# Number of GPUs you'd like to evaluate on. Set this equal to number of experts you'd like to mix.
num_gpus=$1
# Path to data-bins
data_path=$2
# model ensemble, separated by `:` e.g. model_path/checkpoint_last-rank-0.pt:model_path/checkpoint_last-rank-4.pt
model=$3
# target domain to evaluate on
target_domain=$4
# path to posterior calculation output
results_path=$5
# set to "estimate" if you'd like to estimate a prior on validation set (for ensemble_type "cached_prior")
estimate=$6
# comma separated string of per-expert prior, e.g. "0.1,0.2,0.3,0.4". Must be in order experts appear in $model.
precomputed_prior=$7
# Ensemble type, one of "simple_average","cached_prior", "updating_prior", "uniform_prior"
ensemble_type=$8

if [[ $estimate == *"estimate"* ]]; then
	echo "estimating probabilities..."
	target_eval_split=valid_${target_domain};
   srun --label python fairseq_cli/ensemble_eval_lm.py $data_bin \
    --path $model \
    --gen-subset $target_eval_split \
    --target-domain train_${target_domain}\
    --target-eval ${target_eval_split} \
    --task multidomain_language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 128      \
    --batch-size 2  \
    --optimizer adafactor \
    --sample-break-mode none     \
    --log-format simple     \
    --log-interval 50     \
    --skip-invalid-size-inputs-valid-test               \
    --criterion cross_entropy     \
    --lr 5e-4        \
    --weight-decay 0.1     \
    --update-freq 1 \
    --clip-norm 0.0     \
    --no-save           \
    --bucket-cap-mb 200                       \
    --ddp-backend no_c10d      \
    --arch transformer_lm                 \
    --train-domains ${target_domain} \
    --eval-domains ${target_domain} \
    --log-format tqdm \
    --train-subset train_${target_domain} \
    --partial-load \
    --ensemble-type "updating_prior" \
    --results-path ${results_path} \
    --distributed-world-size $num_gpus \
    --distributed-port 12345 \
    --max-samples 100;
else
	target_eval_split=test_${target_domain};
    srun --label python fairseq_cli/ensemble_eval_lm.py $data_bin \
    --path $model \
    --gen-subset $target_eval_split \
    --target-domain train_${target_domain}\
    --target-eval ${target_eval_split} \
    --task multidomain_language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 128      \
    --batch-size 2  \
    --optimizer adafactor \
    --sample-break-mode none     \
    --log-format simple     \
    --log-interval 50     \
    --skip-invalid-size-inputs-valid-test               \
    --criterion cross_entropy     \
    --lr 5e-4        \
    --weight-decay 0.1     \
    --update-freq 1 \
    --clip-norm 0.0     \
    --no-save           \
    --bucket-cap-mb 200                       \
    --ddp-backend no_c10d      \
    --arch transformer_lm                 \
    --train-domains ${target_domain} \
    --eval-domains ${target_domain} \
    --log-format tqdm \
    --train-subset train_${target_domain} \
    --partial-load \
    --results-path ${results_path} \
    --ensemble-type ${ensemble_type} \
    --distributed-world-size $num_gpus \
    --distributed-port 12345 \
    --precomputed-prior ${precomputed_prior}
fi;
