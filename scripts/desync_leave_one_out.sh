target_domain="webtext"
domains_loo="1b,biomed,cs,legal,realnews,reviews,tweets"

function join_by { local d=${1-} f=${2-}; if shift 2; then printf %s "$f" "${@/#/$d}"; fi; }


echo "training on ${domains_loo}"
srun --label python fairseq_cli/train.py     /private/home/suching/data_proc/data-bin/ \
     --task multidomain_language_modeling \
     --sample-break-mode none \
     --log-format simple  \
     --log-interval 50     \
     --skip-invalid-size-inputs-valid-test     \
     --validate-interval-updates 5000     \
     --save-interval-updates 500     \
     --keep-interval-updates 1    \
     --arch transformer_lm    \
     --criterion cross_entropy     \
     --lr-scheduler inverse_sqrt     \
     --warmup-init-lr 1e-07     \
     --lr 0.001               \
     --tokens-per-sample 512          \
     --weight-decay 0.01           \
     --dropout 0.1  \
     --optimizer adam \
     --adam-betas '(0.9, 0.98)'  \
     --clip-norm 0.0      \
     --max-update 100000     \
     --warmup-updates 8000     \
     --update-freq 1     \
     --clip-norm 0.0     \
     --save-dir data_parallel_loo/data_parallel_loo_${target_domain}        \
     --batch-size-valid 2                        \
     --wandb-project modular_lm           \
     --train-domains $domains_loo \
     --eval-domains $domains_loo \
     --disable-validation \
     --batch-size 16 \
     --fp16 \
     --fp16-no-flatten-grads \
     --distributed-world-size 8 \
     --distributed-port 12345 \
     --ddp-backend no_c10d
