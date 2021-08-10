# Path to data binary
DATA_PATH=$1
# Target domain to DAPT to
DOMAIN=$2
# Path to model
MODEL_PATH=$3
# Baseline type: either "full" (update all parameters during DAPT) or "feedforward" (update only feedforward network during DAPT)
EXPERIMENT=$4
# output of dapt'ed model
SERIALIZATION_DIR=$5
# suffix to append to output model path, e.g. "final" or "test"
FILE_SUFFIX=$6

# number of GPUs to train with, we default to eight GPUs
NUM_GPUS=8
# distributed port
PORT=12345


domains=${DOMAIN};
train_subset=train_${DOMAIN};
valid_subset=valid_${DOMAIN};
# wandb project name (to track experiment on wandb.ai)
WANDB_PROJECT=gpt3_experiments

if [[ $MODEL_PATH == *"gpt3_small"* ]]; then
    ARCH=transformer_lm_gpt3_small;
    if [[ $MODEL_PATH == *"demix"* ]]; then
        NUM_STEPS=1000;
        SAVE_INTERVAL_UPDATES=200;
    elif [[ $MODEL_PATH == *"dense"* ]]; then
        NUM_STEPS=750;
        SAVE_INTERVAL_UPDATES=200;
    fi
elif [[ $MODEL_PATH == *"gpt3_med"* ]]; then
    ARCH=transformer_lm_gpt3_medium;
    if [[ $MODEL_PATH == *"demix"* ]]; then
        NUM_STEPS=1000;
        SAVE_INTERVAL_UPDATES=200;
    elif [[ $MODEL_PATH == *"dense"* ]]; then
        NUM_STEPS=750;
        SAVE_INTERVAL_UPDATES=200;
    fi
elif [[ $MODEL_PATH == *"gpt3_large"* ]]; then
    ARCH=transformer_lm_gpt3_large;
    if [[ $MODEL_PATH == *"demix"* ]]; then
        NUM_STEPS=500;
        SAVE_INTERVAL_UPDATES=200;
    elif [[ $MODEL_PATH == *"dense"* ]]; then
        NUM_STEPS=300;
        SAVE_INTERVAL_UPDATES=200;
    fi;
elif [[ $MODEL_PATH == *"gpt3_xl"* ]]; then
    ARCH=transformer_lm_gpt3_xl;
    if [[ $MODEL_PATH == *"demix"* ]]; then
        NUM_STEPS=1250;
        SAVE_INTERVAL_UPDATES=250;
    elif [[ $MODEL_PATH == *"dense"* ]]; then
        NUM_STEPS=1000;
        SAVE_INTERVAL_UPDATES=200;
    fi;
elif [[ $MODEL_PATH == *"transformer_lm"* ]]; then
    ARCH=transformer_lm;
    if [[ $MODEL_PATH == *"demix"* ]]; then
        NUM_STEPS=1000;
        SAVE_INTERVAL_UPDATES=200;
    elif [[ $MODEL_PATH == *"dense"* ]]; then
        NUM_STEPS=750;
        SAVE_INTERVAL_UPDATES=200;
    fi
fi;

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
VALIDATION_INTERVAL=500;
KEEP_INTERVAL_UPDATES=10;

# we reduce LR by 10x during DAPT phase.
LR=5e-5;

CLIP_NORM=0.1;
UPDATE_FREQ=8;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));


if [[ $MODEL_PATH == *"demix"* ]]; then
   if [[ $EXPERIMENT == *"feedforward"* ]]; then
        srun --label python fairseq_cli/train.py     \
                        $DATA_PATH     \
                        --task multidomain_language_modeling     \
                        --sample-break-mode none     \
                        --log-format simple     \
                        --log-interval $LOG_INTERVAL    \
                        --skip-invalid-size-inputs-valid-test     \
                        --validate-interval-updates $VALIDATION_INTERVAL     \
                        --save-interval-updates $SAVE_INTERVAL_UPDATES     \
                        --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
                        --no-epoch-checkpoints \
                        --arch $ARCH    \
                        --criterion cross_entropy    \
                        --lr-scheduler polynomial_decay     \
                        --lr $LR             \
                        --tokens-per-sample $TOKENS_PER_SAMPLE          \
                        --optimizer adam \
                        --adam-betas '(0.9, 0.95)'  \
                        --adam-eps 10e-8 \
                        --weight-decay 0.1 \
                        --num-workers 2 \
                        --max-sentences 2 \
                        --max-sentences-valid 2 \
                        --clip-norm $CLIP_NORM      \
                        --max-update $NUM_STEPS     \
                        --total-num-update $NUM_STEPS     \
                        --warmup-updates $NUM_WARMUP_STEPS     \
                        --wandb-project $WANDB_PROJECT \
                        --save-dir ${SERIALIZATION_DIR}        \
                        --train-subset $train_subset \
                        --valid-subset $valid_subset \
                        --train-domains $domains  \
                        --eval-domains $domains \
                        --required-batch-size-multiple 1 \
                        --update-freq $UPDATE_FREQ \
                        --dropout 0.0 \
                        --finetune-from-model $MODEL_PATH \
                        --desynchronize \
                        --untie-parameters feedforward \
                        --data-parallel-groups "0,1,2,3,4,5,6,7" \
                        --sync-type manual \
                        --distributed-world-size 8 \
                        --distributed-port $PORT \
                        --memory-efficient-fp16 \
                        --adaptation \
			--unbalanced;
        elif [[ $EXPERIMENT == *"full"* ]]; then
                 srun --label python fairseq_cli/train.py     \
                        $DATA_PATH     \
                        --task multidomain_language_modeling     \
                        --sample-break-mode none     \
                        --log-format simple     \
                        --log-interval $LOG_INTERVAL    \
                        --skip-invalid-size-inputs-valid-test     \
                        --validate-interval-updates $VALIDATION_INTERVAL     \
                        --save-interval-updates $SAVE_INTERVAL_UPDATES     \
                        --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
                        --no-epoch-checkpoints \
                        --arch $ARCH    \
                        --criterion cross_entropy    \
                        --lr-scheduler polynomial_decay     \
                        --lr $LR             \
                        --tokens-per-sample $TOKENS_PER_SAMPLE          \
                        --optimizer adam \
                        --adam-betas '(0.9, 0.95)'  \
                        --adam-eps 10e-8 \
                        --weight-decay 0.1 \
                        --num-workers 2 \
                        --max-sentences 2 \
                        --max-sentences-valid 2 \
                        --clip-norm $CLIP_NORM      \
                        --max-update $NUM_STEPS     \
                        --total-num-update $NUM_STEPS     \
                        --warmup-updates $NUM_WARMUP_STEPS     \
                        --wandb-project $WANDB_PROJECT \
                        --save-dir ${SERIALIZATION_DIR}        \
                        --train-subset $train_subset \
                        --valid-subset $valid_subset \
                        --train-domains $domains  \
                        --eval-domains $domains \
                        --required-batch-size-multiple 1 \
                        --update-freq $UPDATE_FREQ \
                        --dropout 0.0 \
                        --finetune-from-model $MODEL_PATH \
                        --desynchronize \
                        --untie-parameters feedforward \
                        --data-parallel-groups "0,1,2,3,4,5,6,7" \
                        --sync-type manual \
                        --distributed-world-size 8 \
                        --distributed-port $PORT \
                        --memory-efficient-fp16 \
			            --unbalanced;
        fi
elif [[ $MODEL_PATH == *"dense"* ]]; then
        if [[ $EXPERIMENT == *"feedforward" ]]; then
        srun --label python fairseq_cli/train.py     \
                        $DATA_PATH     \
                        --task multidomain_language_modeling     \
                        --sample-break-mode none     \
                        --log-format simple     \
                        --log-interval $LOG_INTERVAL    \
                        --skip-invalid-size-inputs-valid-test     \
                        --validate-interval-updates $VALIDATION_INTERVAL     \
                        --save-interval-updates $SAVE_INTERVAL_UPDATES     \
                        --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
                        --no-epoch-checkpoints \
                        --arch $ARCH    \
                        --criterion cross_entropy    \
                        --lr-scheduler polynomial_decay     \
                        --lr $LR             \
                        --tokens-per-sample $TOKENS_PER_SAMPLE          \
                        --optimizer adam \
                        --adam-betas '(0.9, 0.95)'  \
                        --adam-eps 10e-8 \
                        --weight-decay 0.1 \
                        --num-workers 2 \
                        --max-sentences 2 \
                        --max-sentences-valid 2 \
                        --clip-norm $CLIP_NORM      \
                        --max-update $NUM_STEPS     \
                        --total-num-update $NUM_STEPS     \
                        --warmup-updates $NUM_WARMUP_STEPS     \
                        --wandb-project $WANDB_PROJECT \
                        --save-dir ${SERIALIZATION_DIR}        \
                        --train-subset $train_subset \
                        --valid-subset $valid_subset \
                        --train-domains $domains  \
                        --eval-domains $domains \
                        --required-batch-size-multiple 1 \
                        --update-freq $UPDATE_FREQ \
                        --dropout 0.0 \
                        --finetune-from-model $MODEL_PATH \
                        --distributed-world-size 8 \
                        --distributed-port $PORT \
                        --memory-efficient-fp16 \
                        --adaptation \
			            --unbalanced;
        elif [[ $EXPERIMENT == *"full"* ]]; then
                srun --label python fairseq_cli/train.py     \
                        $DATA_PATH     \
                        --task multidomain_language_modeling     \
                        --sample-break-mode none     \
                        --log-format simple     \
                        --log-interval $LOG_INTERVAL    \
                        --skip-invalid-size-inputs-valid-test     \
                        --validate-interval-updates $VALIDATION_INTERVAL     \
                        --save-interval-updates $SAVE_INTERVAL_UPDATES     \
                        --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
                        --no-epoch-checkpoints \
                        --arch $ARCH    \
                        --criterion cross_entropy    \
                        --lr-scheduler polynomial_decay     \
                        --lr $LR             \
                        --tokens-per-sample $TOKENS_PER_SAMPLE          \
                        --optimizer adam \
                        --adam-betas '(0.9, 0.95)'  \
                        --adam-eps 10e-8 \
                        --weight-decay 0.1 \
                        --num-workers 2 \
                        --max-sentences 2 \
                        --max-sentences-valid 2 \
                        --clip-norm $CLIP_NORM      \
                        --max-update $NUM_STEPS     \
                        --total-num-update $NUM_STEPS     \
                        --warmup-updates $NUM_WARMUP_STEPS     \
                        --wandb-project $WANDB_PROJECT \
                        --save-dir ${SERIALIZATION_DIR}        \
                        --train-subset $train_subset \
                        --valid-subset $valid_subset \
                        --train-domains $domains  \
                        --eval-domains $domains \
                        --required-batch-size-multiple 1 \
                        --update-freq $UPDATE_FREQ \
                        --dropout 0.0 \
                        --finetune-from-model $MODEL_PATH \
                        --distributed-world-size 8 \
                        --memory-efficient-fp16 \
                        --distributed-port $PORT \
			            --unbalanced;
        fi
fi
