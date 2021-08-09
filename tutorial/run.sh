# Number of GPUs you'd like to train on
NUM_GPUS=$1
# Number of nodes you'd like to train on (assuming 8 GPUs per node)
NUM_NODES=$((${NUM_GPUS}/8))
# Distributed port
PORT=$2
# Fairseq model name (e.g. transformer_lm; see https://github.com/kernelmachine/demix/blob/main/fairseq/models/transformer_lm.py for other options)
ARCH=$3
# Baseline type: choice between demix, dense, unbalanced_dense, and domain_token
EXPERIMENT=$4
# Path to data-bins
DATA_PATH=$5
# path to directory to where you'd like to output the model
SERIALIZATION_DIR=$6
# suffix to append to model output (e.g. "test", "final")
FILE_SUFFIX=$7


# list of domains you'd like to train on, that can be found in $DATA_PATH
domains=ag_news,amazon,chemprot,citation_intent,hp-news,imdb,rct,1b_test;
# validation datasets for each domain
valid_subset=valid_ag_news,valid_amazon,valid_chemprot,valid_citation_intent,valid_hp-news,valid_imdb,valid_rct,valid_1b_test;

# name of wandb project to track model output (at wandb.ai)
WANDB_PROJECT=debug;

BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=1;

# For the tutorial, we train on smaller sequence lengths.
TOKENS_PER_SAMPLE=128;
LR=5e-4;
CLIP_NORM=0.1;
UPDATE_FREQ=8;
NUM_STEPS=5000;
SAVE_INTERVAL_UPDATES=5000;
VALIDATION_INTERVAL=500;
NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));


# $DATA_PARALLEL_GROUPS identifies which ranks we will synchronize over. "A,B C,D" means we will synchronize ranks A,B and synchronize ranks C,D.
if [[ $EXPERIMENT == *"dense"*  || $EXPERIMENT == *"domain_token"* ]]; then
     DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7";
elif  [[ $EXPERIMENT == *"demix"* ]]; then
     DATA_PARALLEL_GROUPS="0 1 2 3 4 5 6 7";
fi;

if [[ $EXPERIMENT == *"demix"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
          --task multidomain_language_modeling \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
	     --criterion desynchronized_cross_entropy     \
          --lr-scheduler polynomial_decay     \
          --num-workers 2 \
          --max-sentences $BATCH_SIZE \
          --no-epoch-checkpoints \
          --max-sentences-valid $BATCH_SIZE \
          --lr $LR              \
          --tokens-per-sample $TOKENS_PER_SAMPLE          \
          --optimizer adam \
          --adam-betas '(0.9, 0.95)'  \
          --adam-eps 10e-8 \
          --weight-decay 0.1 \
          --clip-norm $CLIP_NORM      \
          --max-update $NUM_STEPS     \
          --total-num-update $NUM_STEPS     \
          --warmup-updates $NUM_WARMUP_STEPS     \
          --update-freq $UPDATE_FREQ     \
          --save-dir ${SERIALIZATION_DIR}/demix_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --valid-subset $valid_subset \
          --train-domains $domains  \
          --eval-domains $domains \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          --distributed-world-size $NUM_GPUS \
          --distributed-port $PORT \
          --desynchronize --domain-parallel \
          --ddp-backend no_c10d \
          --sync-type manual \
          --untie-parameters feedforward \
          --data-parallel-groups "${DATA_PARALLEL_GROUPS}" \
          --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"unbalanced"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
               --task multidomain_language_modeling \
               --sample-break-mode none \
               --log-format simple  \
               --log-interval $LOG_INTERVAL    \
               --skip-invalid-size-inputs-valid-test     \
               --validate-interval-updates $VALIDATION_INTERVAL     \
               --save-interval-updates $SAVE_INTERVAL_UPDATES     \
               --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
               --arch $ARCH    \
               --criterion desynchronized_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --no-epoch-checkpoints \
               --max-sentences-valid $BATCH_SIZE \
               --lr $LR              \
               --tokens-per-sample $TOKENS_PER_SAMPLE          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --update-freq $UPDATE_FREQ     \
               --save-dir ${SERIALIZATION_DIR}/unbalanced_dense_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
               --batch-size-valid 2                        \
               --wandb-project $WANDB_PROJECT           \
               --valid-subset $valid_subset \
               --train-domains $domains  \
               --eval-domains $domains \
               --required-batch-size-multiple 1 \
               --memory-efficient-fp16 \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --all-gather-list-size 32000 \
               --unbalanced;
elif [[ $EXPERIMENT == *"dense"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
               --task multidomain_language_modeling \
               --sample-break-mode none \
               --log-format simple  \
               --log-interval $LOG_INTERVAL    \
               --skip-invalid-size-inputs-valid-test     \
               --validate-interval-updates $VALIDATION_INTERVAL     \
               --save-interval-updates $SAVE_INTERVAL_UPDATES     \
               --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
               --arch $ARCH    \
               --criterion desynchronized_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --no-epoch-checkpoints \
               --max-sentences-valid $BATCH_SIZE \
               --lr $LR              \
               --tokens-per-sample $TOKENS_PER_SAMPLE          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --update-freq $UPDATE_FREQ     \
               --save-dir ${SERIALIZATION_DIR}/dense_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
               --batch-size-valid 2                        \
               --wandb-project $WANDB_PROJECT           \
               --valid-subset $valid_subset \
               --train-domains $domains  \
               --eval-domains $domains \
               --required-batch-size-multiple 1 \
               --memory-efficient-fp16 \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"switch"* ]]; then
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
               --criterion moe_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --lr $LR              \
               --tokens-per-sample 1024          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --max-sentences-valid $BATCH_SIZE \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --wandb-project $WANDB_PROJECT \
               --save-dir ${SERIALIZATION_DIR}/switch_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}         \
               --batch-size-valid 2                        \
               --train-domains $domains \
               --eval-domains $domains \
               --valid-subset $valid_subset \
               --required-batch-size-multiple 1 \
               --update-freq $UPDATE_FREQ \
               --fp16 \
               --fp16-no-flatten-grads \
               --moe-freq 2 \
               --moe-top1-expert \
               --moe-expert-count $NUM_GPUS \
               --moe-gating-use-fp32 \
               --moe-gate-loss-wt 0.01 \
               --moe-gate-loss-combine-method sum \
               --moe-second-expert-policy all \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --ddp-backend no_c10d \
               --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"gshard"* ]]; then
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
               --criterion moe_cross_entropy     \
               --lr-scheduler polynomial_decay     \
               --lr $LR               \
               --tokens-per-sample 1024          \
               --optimizer adam \
               --adam-betas '(0.9, 0.95)'  \
               --adam-eps 10e-8 \
               --weight-decay 0.1 \
               --num-workers 2 \
               --max-sentences $BATCH_SIZE \
               --max-sentences-valid $BATCH_SIZE \
               --clip-norm $CLIP_NORM      \
               --max-update $NUM_STEPS     \
               --total-num-update $NUM_STEPS     \
               --warmup-updates $NUM_WARMUP_STEPS     \
               --wandb-project $WANDB_PROJECT \
               --save-dir ${SERIALIZATION_DIR}/gshard_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}         \
               --batch-size-valid 2                        \
               --train-domains $domains \
               --eval-domains $domains \
               --valid-subset $valid_subset \
               --required-batch-size-multiple 1 \
               --update-freq $UPDATE_FREQ \
               --fp16 \
               --fp16-no-flatten-grads \
               --moe-freq 2 \
               --moe-expert-count $NUM_GPUS \
               --moe-gating-use-fp32 \
               --moe-gate-loss-wt 0.01 \
               --moe-gate-loss-combine-method sum \
               --moe-second-expert-policy all \
               --distributed-world-size $NUM_GPUS \
               --distributed-port $PORT \
               --ddp-backend no_c10d \
               --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"domain_token"* ]]; then
     srun --label python fairseq_cli/train.py     $DATA_PATH \
          --task multidomain_language_modeling \
          --sample-break-mode none \
          --log-format simple  \
          --log-interval $LOG_INTERVAL    \
          --skip-invalid-size-inputs-valid-test     \
          --validate-interval-updates $VALIDATION_INTERVAL     \
          --save-interval-updates $SAVE_INTERVAL_UPDATES     \
          --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
          --arch $ARCH    \
	     --criterion desynchronized_cross_entropy     \
          --lr-scheduler polynomial_decay     \
          --num-workers 2 \
          --max-sentences $BATCH_SIZE \
          --no-epoch-checkpoints \
          --max-sentences-valid $BATCH_SIZE \
          --lr $LR              \
          --tokens-per-sample $TOKENS_PER_SAMPLE          \
          --optimizer adam \
          --adam-betas '(0.9, 0.95)'  \
          --adam-eps 10e-8 \
          --weight-decay 0.1 \
          --clip-norm $CLIP_NORM      \
          --max-update $NUM_STEPS     \
          --total-num-update $NUM_STEPS     \
          --warmup-updates $NUM_WARMUP_STEPS     \
          --update-freq $UPDATE_FREQ     \
          --save-dir ${SERIALIZATION_DIR}/domain_token_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
          --batch-size-valid 2                        \
          --wandb-project $WANDB_PROJECT           \
          --valid-subset $valid_subset \
          --train-domains $domains  \
          --eval-domains $domains \
          --required-batch-size-multiple 1 \
          --memory-efficient-fp16 \
          --distributed-world-size $NUM_GPUS \
          --distributed-port $PORT \
          --all-gather-list-size 32000 \
          --add-domain-token;
fi;
