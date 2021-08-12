# DEMix

This repository contains modeling utilities for "DEMix Layers: Disentangling Domains for Modular Language Modeling" (Gururangan et. al, 2021).

This code is a fork of Fairseq. It is based on Python 3.8, CUDA 11 and includes PyTorch 1.8.0, NCCL 2.8.4 and apex.

## Dataset

The multidomain dataset scripts are housed in another repository, located [here](https://github.com/kernelmachine/demix-data). Clone that repository and follow instructions to setup data to train on.

Follow that tutorial to generate data-bins on eight (small) example domains.

Make sure to set the `DATA_DIR` accordingly.


## Fairseq Installation

If you've already made an environment from the dataset creation phase, just use that. Otherwise:

```bash
conda create env --name demix
cd demix/
pip install --editable .
```

Additionally, please make sure you have the dependencies above installed (check Fairseq documentation for more information).


## Tutorial

Here we will follow a tutorial to train on the example domains from the tutorial in the DEMix-data repository.  Note that the model that results from this tutorial is pretty bad, because we're working with very small amounts of data and also a small LM. This tutorial is there to help you quickly understand the pipeline, and ensure that each script completes successfully.

To replicate the DEMix paper, with a GPT-3 model, follow the instructions [here](REPLICATE_PAPER.sh).

### Basic Training

After setting up the example domains, run the following to train a small language model. Note that the scripts in this paper assume you are running on a multi-node GPU cluster with SLURM.


First, allocate some nodes, with GPUs with at least 32GB of RAM. Here we allocate 1 node with 8 volta32GB GPUs.


```bash

salloc --gpus-per-node 8 --nodes 1  -C 'volta32gb' --ntasks-per-node 8 --cpus-per-task 10 --mem 400G --time XXX --partition YYY
```

Then run:

```bash
export NUM_GPUS=8
export DISTRIBUTED_PORT=12345
export MODEL=transformer_lm
export EXPERIMENT=demix
# $DATA_DIR was set in DEMix-data tutorial.
export DATA_BIN=${DATA_DIR}/data-bin/
export EXPERIMENT_SUFFIX=tutorial
export SERIALIZATION_DIR=$(pwd)/demix_tutorial_model
bash tutorial/train.sh $NUM_GPUS \
                    $DISTRIBUTED_PORT \
                    $MODEL \
                    $EXPERIMENT \
                    $DATA_BIN \
                    $SERIALIZATION_DIR \
                    $EXPERIMENT_SUFFIX
```

This will output a trained language model in `${SERIALIZATION_DIR}`


To train balanced dense LM, set `export EXPERIMENT=dense`, to train unbalanced dense LM, set `export EXPERIMENT=unbalanced`, to train "+Domain Token" LM , set `export EXPERIMENT=domain_token`.

We have provided a simple script `demix/train.sh`, with the same interface, with all hyperparameter preset to help replicate results in the paper.

## Evaluation

We have two ways to evaluate the demix language model: with and without mixing experts.

### Evaluating without mixing experts

To evaluate the language model _without_ mixing experts, you can supply the checkpoint from a GPU on a particular rank (to specify the use of the domain expert that was trained on that GPU):

```bash
export DATA_BIN=${DATA_DIR}/data-bin/
export GPU_RANK=0
export PATH_TO_CHECKPOINT=${SERIALIZATION_DIR}/checkpoint_last-rank-${GPU_RANK}.pt
export OUTPUT_PATH=eval_output.jsonl
export SPLIT=valid
export DOMAIN=imdb
bash tutorial/eval_lm.sh $DATA_BIN $PATH_TO_CHECKPOINT $OUTPUT_PATH $SPLIT $DOMAIN
```

To evaluate on test data, set `export SPLIT=test`

The same script is used for the other baselines.


For the +domain token model, you can additionally supply a domain token to use at test time:

```bash
export DOMAIN_TOKEN=XXX
bash tutorial/eval_lm.sh $DATA_BIN $PATH_TO_CHECKPOINT $OUTPUT_PATH $SPLIT $DOMAIN $DOMAIN_TOKEN
```

### Evaluating with mixing experts

First, we estimate the posterior distribution on 100 sequences of validation data of the domain using the following command:

```bash
export DATA_BIN=${DATA_DIR}/data-bin
export DOMAIN=imdb
export DEV_POSTERIOR_OUTPUT=dev_posteriors.jsonl
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
export NUM_EVALUATION_GPUS=8;
bash tutorial/mix_eval_lm.sh $NUM_EVALUATION_GPUS $DATA_BIN  ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-7.pt $DOMAIN $DEV_POSTERIOR_OUTPUT estimate;
```

Then, we open `$POSTERIOR_OUTPUT`, extracting the `exp_avg_posterior` value of the last line in that file:


```bash
export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')
```

We use this posterior as the domain prior (supplied as a string) when evaluating on test data, like so:

```bash
bash tutorial/mix_eval_lm.sh $NUM_EVALUATION_GPUS $DATA_BIN  ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-7.pt $DOMAIN $DEV_POSTERIOR_OUTPUT eval $POSTERIOR cached_prior;
```

## Adapting the Language Model

We additionally provide scripts to adapt the language model to a new domain.


### DEMix DAPT
In this tutorial, we just adapt one of the existing experts to a new example domain in the `demix-data` project, located in `/path/to/demix-data/new_example_domains`.

First, we need to figure out which domain expert has the most affinity to the target domain we want to adapt to:

```bash
export NEW_DATA_BIN=/private/home/suching/demix-data/new_example_domains/data-bin/
export NEW_DOMAIN=acl_papers
export DEV_POSTERIOR_OUTPUT=${NEW_DOMAIN}_posterior.jsonl
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
export NUM_EVALUATION_GPUS=8;
bash tutorial/mix_eval_lm.sh $NUM_EVALUATION_GPUS $NEW_DATA_BIN  ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-7.pt $NEW_DOMAIN $DEV_POSTERIOR_OUTPUT estimate;
export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')
```

Here, we find that the most likely expert is expert number 5.

```bash
export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')
echo $POSTERIOR
```

We then adapt expert 5 to the target domain using the `tutorial/dapt.sh` script, using DEMix DAPT:

```bash
export PATH_TO_CHECKPOINT=${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt
export UNFREEZE_PARAMETERS=feedforward
export NEW_SERIALIZATION_DIR=$(pwd)/${NEW_DOMAIN}_demix_dapt
export EXPERIMENT_SUFFIX=test
bash tutorial/dapt.sh $NEW_DATA_BIN $NEW_DOMAIN $PATH_TO_CHECKPOINT $UNFREEZE_PARAMETERS $NEW_SERIALIZATION_DIR $EXPERIMENT_SUFFIX
```

Once this is trained, you can add that expert to your ensemble when evaluating on new data:


```bash
export NEW_DATA_BIN=/path/to/demix-data/new_example_domains/data-bin/
export NEW_DOMAIN=acl_papers
export DEV_POSTERIOR_OUTPUT=${NEW_DOMAIN}_posterior.jsonl
# set NUM_EVALUATION_GPUS equal to the number of experts you'd like to ensemble.
export NUM_EVALUATION_GPUS=8;
export PATH_TO_NEW_EXPERT=${NEW_SERIALIZATION_DIR}/checkpoint_last-rank-0.pt
bash tutorial/mix_eval_lm.sh $NUM_EVALUATION_GPUS $NEW_DATA_BIN  ${SERIALIZATION_DIR}/checkpoint_last-rank-0.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-1.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-2.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-3.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-4.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-5.pt:${SERIALIZATION_DIR}/checkpoint_last-rank-6.pt:${PATH_TO_NEW_EXPERT} $NEW_DOMAIN $DEV_POSTERIOR_OUTPUT estimate;
export POSTERIOR=$(tail -n 1 $DEV_POSTERIOR_OUTPUT | jq -rc '.exp_avg_posterior | join(",")')
```


### Dense DAPT

If you wanted to do Dense DAPT instead, just change the environment variables:

```bash
export PATH_TO_CHECKPOINT=/path/to/dense/model/checkpoint_last.pt
export FEEDFORWARD_OR_FULL=full
export SERIALIZATION_DIR=$(pwd)/${NEW_DOMAIN}_dense_dapt
export EXPERIMENT_SUFFIX=test
bash tutorial/dapt.sh $NEW_DATA_BIN $NEW_DOMAIN $PATH_TO_CHECKPOINT $FEEDFORWARD_OR_FULL $SERIALIZATION_DIR $EXPERIMENT_SUFFIX
```
