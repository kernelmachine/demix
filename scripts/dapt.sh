DATA_PATH=$1
DOMAIN=$2
MODEL_PATH=$3
EXPERIMENT=$4
SERIALIZATION_DIR=$5
FILE_SUFFIX=$6
NUM_GPUS=8
PORT=20555
#salloc --gpus-per-node 8 --nodes $NUM_NODES  --ntasks-per-node 8 --cpus-per-task 10 --mem 480G --time 3000 --partition learnfair,dev

domains=${DOMAIN};
train_subset=train_${DOMAIN};
valid_subset=valid_${DOMAIN};
WANDB_PROJECT=gpt3_experiments

ARCH=transformer_lm_gpt3_medium
TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
VALIDATION_INTERVAL=500;
KEEP_INTERVAL_UPDATES=10;

LR=5e-5;
CLIP_NORM=0.1;
UPDATE_FREQ=8;
if [[ $MODEL_PATH == *"domain_parallel"* ]]; then 
	NUM_STEPS=100;
elif [[ $MODEL_PATH == *"data_parallel"* ]]; then
	NUM_STEPS=750;
fi

SAVE_INTERVAL_UPDATES=1000;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));


if [[ $MODEL_PATH == *"domain_parallel"* ]]; then
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
                        --distributed-port 1234 \
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
                        --distributed-port 1234 \
			--unbalanced;
        fi
elif [[ $MODEL_PATH == *"data_parallel"* ]]; then
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
                        --distributed-port 1234 \
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
                        --distributed-port 1234 \
			--unbalanced;
        fi
fi
