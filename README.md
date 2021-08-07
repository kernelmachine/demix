# DEMix Layers
DEMix Layers for Modular Language Modeling


This code is a fork of Fairseq. It is based on Python 3.8, CUDA 11 and includes PyTorch 1.8.0, NCCL 2.8.4 and apex.

## Installation

```bash
conda create env demix
cd demix/
pip install --editable .
```

Additionally, please make sure you have the dependencies above installed (check Fairseq documentation for more information).

## Dataset


The multidomain dataset scripts are housed in another repository, located [here](https://github.com/kernelmachine/demix-data). Clone that repository and follow instructions to setup data to train on.

## Basic Training

We have provided a simple script `scripts/run_all.sh` with all hyperparameter preset to help replicate results in the paper. Note that training assumes you are running on a multi-node GPU cluster with SLURM.

First, allocate some nodes, with GPUs with at least 32GB of RAM. Here we allocate 4 nodes with 8 GPUs per node, so 32 GPUs in total.

```bash
salloc --gpus-per-node 8 --nodes 4  -C 'volta32gb' --ntasks-per-node 8 --cpus-per-task 10 --mem 400G --time XXX --partition YYY
```

The `scripts/run_all.sh` follows a common format:

```bash
export NUM_GPUS=32
export DISTRIBUTED_PORT=12345
export MODEL=transformer_lm_gpt3_small
export EXPERIMENT=demix
export DATA_PATH=/path/to/multidomain/data/
export EXPERIMENT_SUFFIX=test
export SERIALIZATION_DIR=/path/to/serialization/dir
bash scripts/run_all.sh $NUM_GPUS $DISTRIBUTED_PORT $MODEL $EXPERIMENT $DATA_PATH $SERIALIZATION_DIR $EXPERIMENT_SUFFIX
```

To train balanced dense LM, set `export EXPERIMENT=dense`, to train unbalanced dense LM, set `export EXPERIMENT=unbalanced`, to train +domain token LM , set `export EXPERIMENT=domain_token`.

## Evaluation

We have two ways to evaluate the demix language model, with and without mixing experts.

### Evaluating without mixing experts

To evaluate the language model _without_ mixing experts:

```bash
export DATA_PATH=/path/to/multidomain/data/
export PATH_TO_CHECKPOINT=/path/to/serialization/dir/checkpoint_last.pt
export OUTPUT_PATH=eval_output.jsonl
export SPLIT=valid
export DOMAIN=XXX
export DOMAIN_TOKEN=XXX
bash scripts/eval_lm_single.sh $DATA_PATH $PATH_TO_CHECKPOINT $OUTPUT_PATH $SPLIT $DOMAIN $DOMAIN_TOKEN
```

To evaluate on test data, set `export SPLIT=test`

The same script is used for the other baselines.

For + domain token, you can additionally supply a domain token to use at test time:

```bash
export DOMAIN_TOKEN=XXX
bash scripts/eval_lm_single.sh $DATA_PATH $PATH_TO_CHECKPOINT $OUTPUT_PATH $SPLIT $DOMAIN $DOMAIN_TOKEN
```

For demix mdoel, you can supply the checkpoint from a GPU on a particular rank (to specify the use of a specific domain expert)

```bash
export PATH_TO_CHECKPOINT=/path/to/serialization/dir/checkpoint_last-rank-X.pt
bash scripts/eval_lm_single.sh $DATA_PATH $PATH_TO_CHECKPOINT $OUTPUT_PATH $SPLIT $DOMAIN $DOMAIN_TOKEN
```

### Evaluating with mixing experts

First, we estimate the posterior distribution on 100 sequences of validation data of the domain using the following command:

```bash
export DATA_PATH=/path/to/multidomain/data/
export MODEL_NAME=demix_32_GPUs_transformer_lm_gpt3_small_test
export DOMAIN=XXX
bash scripts/mix_experts.sh $DATA_PATH $MODEL_NAME $DOMAIN $DOMAIN $POSTERIOR_OUTPUT estimate
```

Then, we open $POSTERIOR_OUTPUT, copying the `posterior` value of the last line in that file.

We use this posterior as the domain prior (supplied as a string) when evaluating on test data, like so:

```bash
export DATA_PATH=/path/to/multidomain/data/
export MODEL_NAME=demix_32_GPUs_transformer_lm_gpt3_small_test
export DOMAIN=XXX
bash scripts/mix_experts.sh $DATA_PATH $MODEL_NAME $DOMAIN $DOMAIN $POSTERIOR_OUTPUT eval '0.2,0.5,0.3'
```

## Adapting the Language Model

We additionally provide scripts to adapt the language model to a new domain.

```bash
export NEW_DATA_PATH=/path/to/new/domains
export NEW_DOMAIN=XXX
export PATH_TO_CHECKPOINT=/path/to/serialization/dir/checkpoint_last-rank-XXX.pt
export FEEDFORWARD_OR_FULL=feedforward
export SERIALIZATION_DIR=/path/to/new/serialization/dir
export EXPERIMENT_SUFFIX=test
bash scripts/dapt_small.sh $NEW_DATA_PATH NEW_DOMAIN $PATH_TO_CHECKPOINT $FEEDFORWARD_OR_FULL $SERIALIZATION_DIR $EXPERIMENT_SUFFIX
```
