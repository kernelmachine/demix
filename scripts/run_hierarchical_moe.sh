srun --label python fairseq_cli/train.py \
    /private/home/suching/data_proc/data-bin-sample/ \
    --task multidomain_language_modeling \
    --share-decoder-input-output-embed \
    --sample-break-mode none \
    --log-format simple \
    --log-interval 50 \
    --skip-invalid-size-inputs-valid-test \
    --validate-interval-updates 5000 \
    --save-interval-updates 10000 \
    --keep-interval-updates 1 \
    --arch transformer_lm \
    --criterion moe_cross_entropy \
    --moe-gate-loss-wt 0.01 \
    --lr-scheduler inverse_sqrt \
    --warmup-init-lr 0.001 \
    --lr 0.001 \
    --batch-size 16 \
    --min-loss-scale 1e-10 \
    --tokens-per-sample 256 \
    --optimizer adafactor \
    --weight-decay 0.0 \
    --decoder-attention-heads 4  \
    --decoder-layers 4 \
    --decoder-ffn-embed-dim 16384 \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --relu-dropout 0.1 \
    --max-update 100000 \
    --warmup-updates 10000 \
    --update-freq 1 \
    --clip-norm 0.0 \
    --save-dir ./test_switch_hierarchical_8_experts_ffn_test \
    --moe-freq 2 \
    --moe-num-experts-per-domain 1 \
    --moe-num-domains 8 \
    --moe-expert-count 8 \
    --moe-hierarchical-expert \
    --required-batch-size-multiple 16  \
    --batch-size-valid 2  \
    --bucket-cap-mb 200  \
    --moe-expert-decoder-ffn-dim 16384 \
    --fp16-no-flatten-grads   \
    --wandb-project modular_lm  \
    --distributed-world-size 8 \
    --distributed-port 12334   \
    --fp16 \
    --moe-gating-use-fp32 \
    --moe-expert-type ffn \
    --ddp-backend no_c10d \
    # --valid-subset valid_1b,valid_biomed,valid_cs,valid_legal,valid_realnews,valid_reviews,valid_tweets,valid_webtext \
