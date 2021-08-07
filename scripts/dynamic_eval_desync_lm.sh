data_bin=$1
model=$2
target_domain=$3
undomain=$4
results_path=$5


python fairseq_cli/dynamic_eval_lm.py $data_bin \
    --path $model \
    --gen-subset train_${target_domain},valid_${target_domain},valid_${undomain},train_${undomain} \
    --target-domain ${target_domain}\
    --undomain-eval valid_${undomain} \
    --undomain-replay train_${undomain} \
    --task multidomain_language_modeling \
    --sample-break-mode none \
    --tokens-per-sample 256      \
    --batch-size 16  \
    --original-domains 1b,biomed,cs,legal,realnews,reviews,tweets,webtext \
    --optimizer adafactor \
    --sample-break-mode none     \
    --log-format simple     \
    --log-interval 50     \
    --skip-invalid-size-inputs-valid-test               \
    --criterion cross_entropy     \
    --lr 0.001          \
    --min-loss-scale 1e-1              \
    --weight-decay 0.0     \
    --decoder-attention-heads 4 \
    --update-freq 1     \
    --clip-norm 0.0     \
    --no-save           \
    --bucket-cap-mb 200                       \
    --ddp-backend no_c10d      \
    --arch transformer_lm                 \
    --decoder-layers 4     \
    --decoder-ffn-embed-dim 16384     \
    --dropout 0.1     \
    --attention-dropout 0.01 \
    --train-domains ${target_domain} \
    --eval-domains ${undomain} \
    --log-format tqdm \
    --train-subset train_${target_domain} \
    --partial-load \
    --eval-only \
    --results-path ${results_path}\
    --add-expert-layers
