data_bin=$1
model=$2
target_domain=$3
results_path=$4
estimate=$5
precomputed_prior=$6

# Ensemble type, one of "simple_average","cached_prior", "updating_prior", "uniform_prior"
ensemble_type=$7

if [[ $estimate == *"estimate"* ]]; then
	echo "estimating probabilities..."
	target_eval_split=valid_${target_domain};
   python fairseq_cli/ensemble_eval_lm.py $data_bin \
    --path $model \
    --gen-subset $target_eval_split \
    --target-domain train_${target_domain}\
    --target-eval ${target_eval_split} \
    --task multidomain_language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 1024      \
    --batch-size 2  \
    --original-domains 1b,cs,legal,med,anonymized_openwebtext,anonymized_realnews,reddit,anonymized_reviews \
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
    --max-samples 100;
else
	target_eval_split=test_${target_domain};
    python fairseq_cli/ensemble_eval_lm.py $data_bin \
    --path $model \
    --gen-subset $target_eval_split \
    --target-domain train_${target_domain}\
    --target-eval ${target_eval_split} \
    --task multidomain_language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 1024      \
    --batch-size 2  \
    --original-domains 1b,cs,legal,med,anonymized_openwebtext,anonymized_realnews,reddit,anonymized_reviews \
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
    --precomputed-prior ${precomputed_prior}
fi;
