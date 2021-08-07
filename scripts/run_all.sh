NUM_GPUS=$1
NUM_NODES=$((${NUM_GPUS}/8))
PORT=$2
ARCH=$3
EXPERIMENT=$4
DATA_PATH=$5
SERIALIZATION_DIR=$6
FILE_SUFFIX=$7
DEBUG=$8

export SLURM_CONF=/public/slurm/20.11.5/etc/slurm.conf

if [[ $DEBUG == "debug" ]]; then
     domains=1b,cs,legal,biomed,webtext,realnews,tweets,reviews;
     valid_subset=valid_1b,valid_cs,valid_legal,valid_biomed,valid_webtext,valid_realnews,valid_tweets,valid_reviews;
     WANDB_PROJECT=debug
elif [[ $EXPERIMENT == *"cluster"* ]]; then
     domains=0,1,2,3,4,5,6,7;
     valid_subset=valid_0,valid_1,valid_2,valid_3,valid_4,valid_5,valid_6,valid_7;
     WANDB_PROJECT=gpt3_experiments
elif [[ $EXPERIMENT == *"loo"* ]]; then
     domains=1b,cs,legal,anonymized_openwebtext,anonymized_realnews,reddit,anonymized_reviews;
     valid_subset=valid_1b,valid_cs,valid_legal,valid_anonymized_openwebtext,valid_anonymized_realnews,valid_reddit,valid_anonymized_reviews;
     WANDB_PROJECT=gpt3_experiments
else
     # TODO: alphabetize this!
     domains=1b,cs,legal,med,anonymized_openwebtext,anonymized_realnews,reddit,anonymized_reviews;
     valid_subset=valid_1b,valid_cs,valid_legal,valid_med,valid_anonymized_openwebtext,valid_anonymized_realnews,valid_reddit,valid_anonymized_reviews;
     WANDB_PROJECT=gpt3_experiments
fi;

TOKENS_PER_SAMPLE=1024;
BATCH_SIZE=2;
LOG_INTERVAL=50;
KEEP_INTERVAL_UPDATES=1;
if [[ $ARCH == *"gpt3_small"* ]]; then
     LR=5e-4;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
     NUM_STEPS=300000;
     SAVE_INTERVAL_UPDATES=6000;
     VALIDATION_INTERVAL=3000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
elif [[ $ARCH == *"gpt3_medium"* ]]; then
     LR=5e-4;
     NUM_STEPS=120000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=3000;
     VALIDATION_INTERVAL=2000;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
elif [[ $ARCH == *"gpt3_large"* ]]; then
     LR=5e-4;
     NUM_STEPS=65000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=1000;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
elif [[ $ARCH == *"gpt3_xl"* ]]; then
     LR=5e-4;     
     NUM_STEPS=50000;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
     SAVE_INTERVAL_UPDATES=2000;
     VALIDATION_INTERVAL=500;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
elif [[ $ARCH == *"transformer_lm"* ]]; then
     LR=5e-4;
     CLIP_NORM=0.1;
     UPDATE_FREQ=8;
     NUM_STEPS=725000;
     SAVE_INTERVAL_UPDATES=200;
     NUM_WARMUP_STEPS=$((${NUM_STEPS} * 8 / 100));
fi;

if [[ $NUM_GPUS == "8" ]]; then
     if [[ $EXPERIMENT == *"data_parallel"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7";
     elif  [[ $EXPERIMENT == *"domain_parallel"* ]]; then 
          DATA_PARALLEL_GROUPS="0 1 2 3 4 5 6 7";
     fi;
elif [[ $NUM_GPUS == "16" ]]; then
     if [[ $EXPERIMENT == *"data_parallel"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15";
     elif  [[ $EXPERIMENT == *"domain_parallel"* ]]; then 
          DATA_PARALLEL_GROUPS="0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15";
     fi;
elif [[ $NUM_GPUS == "32" ]]; then
     if [[ $EXPERIMENT == *"data_parallel"*  || $EXPERIMENT == *"domain_token"* ]]; then
          DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31";
     elif  [[ $EXPERIMENT == *"domain_parallel"* ]]; then 
          DATA_PARALLEL_GROUPS="0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 16,17,18,19 20,21,22,23 24,25,26,27 28,29,30,31";
     fi;
elif [[ $NUM_GPUS == "64" ]]; then
     if [[ $EXPERIMENT == *"data_parallel"*  || $EXPERIMENT == *"domain_token"* ]]; then
     	DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63";
     elif  [[ $EXPERIMENT == *"domain_parallel"* ]]; then
	     DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63";
     fi;
elif [[ $NUM_GPUS == "128" ]]; then
     if [[ $EXPERIMENT == *"data_parallel"*  || $EXPERIMENT == *"domain_token"* ]]; then
     	DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,65,66,67,68,69,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,90,91,92,93,94,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,110,111,112,113,114,115,116,117,118,119,120,121,122,123,124,125,126,127";
     elif  [[ $EXPERIMENT == *"domain_parallel"* ]]; then
	     DATA_PARALLEL_GROUPS="0,1,2,3,4,5,6,7 8,9,10,11,12,13,14,15 16,17,18,19,20,21,22,23 24,25,26,27,28,29,30,31 32,33,34,35,36,37,38,39 40,41,42,43,44,45,46,47 48,49,50,51,52,53,54,55 56,57,58,59,60,61,62,63 64,65,66,67,68,69,70,71 72,73,74,75,76,77,78,79 80,81,82,83,84,85,86,87 88,89,90,91,92,93,94,95 96,97,98,99,100,101,102,103 104,105,106,107,108,109,110,111 112,113,114,115,116,117,118,119 120,121,122,123,124,125,126,127";
     fi;
fi;


if [[ $EXPERIMENT == "local" ]]; then
     echo "Running locally...";
else
     echo "requesting ${NUM_GPUS} GPUs over ${NUM_NODES} nodes...";
     echo "salloc --gpus-per-node 8 --nodes ${NUM_NODES}  --ntasks-per-node 8 --cpus-per-task 10 --mem 480G --time 3000 --partition learnfair,dev";
fi;
#salloc --gpus-per-node 8 --nodes 1  -C "volta32gb" --ntasks-per-node 8 --cpus-per-task 10 --mem 480G --time 3000 --partition learnfair,dev

if [[ $EXPERIMENT == *"loo"* ]]; then
# None
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
          --save-dir ${SERIALIZATION_DIR}/${EXPERIMENT}_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
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

elif [[ $EXPERIMENT == *"domain_parallel"* ]]; then
# None
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
          --save-dir ${SERIALIZATION_DIR}/domain_parallel_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
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
          --save-dir ${SERIALIZATION_DIR}/unbalanced_data_parallel_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
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
elif [[ $EXPERIMENT == *"data_parallel"* ]]; then

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
          --save-dir ${SERIALIZATION_DIR}/data_parallel_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}      \
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
# data_parallel
# srun --label python fairseq_cli/train.py     \
#             $DATA_PATH     \
#             --task multidomain_language_modeling     \
#             --sample-break-mode none     \
#             --log-format simple     \
#             --log-interval $LOG_INTERVAL    \
#             --skip-invalid-size-inputs-valid-test     \
#             --validate-interval-updates $VALIDATION_INTERVAL     \
#             --save-interval-updates $SAVE_INTERVAL_UPDATES     \
#             --keep-interval-updates $KEEP_INTERVAL_UPDATES     \
#             --no-epoch-checkpoints \
#             --arch $ARCH    \
#             --criterion cross_entropy     \
#             --lr-scheduler polynomial_decay     \
#             --lr $LR             \
#             --tokens-per-sample $TOKENS_PER_SAMPLE          \
#             --optimizer adam \
#             --adam-betas '(0.9, 0.95)'  \
#             --adam-eps 10e-8 \
#             --weight-decay 0.1 \
#             --num-workers 2 \
#             --max-sentences $BATCH_SIZE \
#             --max-sentences-valid $BATCH_SIZE \
#             --clip-norm $CLIP_NORM      \
#             --max-update $NUM_STEPS     \
#             --total-num-update $NUM_STEPS     \
#             --warmup-updates $NUM_WARMUP_STEPS     \
#             --wandb-project $WANDB_PROJECT \
#             --save-dir ${SERIALIZATION_DIR}/data_parallel_${NUM_GPUS}_GPUs_${ARCH}_${FILE_SUFFIX}        \
#             --batch-size-valid 2                        \
# 	       --valid-subset $valid_subset \
#             --train-domains $domains  \
#             --eval-domains $domains \
#             --required-batch-size-multiple 1 \
#             --update-freq $UPDATE_FREQ \
#             --memory-efficient-fp16 \
#             --distributed-world-size $NUM_GPUS \
#             --distributed-port $PORT \
#             --all-gather-list-size 32000;
elif [[ $EXPERIMENT == *"switch"* ]]; then
# data_parallel
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
# data_parallel
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
     # domain token
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


# if [[ $EXPERIMENT == *"loo"* ]]; then
# # None
# srun --label python fairseq_cli/train.py     $DATA_PATH \
#      --task multidomain_language_modeling \
#      --sample-break-mode none \
#      --log-format simple  \
#      --log-interval $LOG_INTERVAL    \
#      --skip-invalid-size-inputs-valid-test     \
#      --validate-interval-updates $VALIDATION_INTERVAL     \
#      --save-interval-updates $SAVE_INTERVAL_UPDATES     \
#      --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
#      --arch $ARCH    \
#      --criterion desynchronized_cross_entropy     \
#      --lr-scheduler polynomial_decay     \
#      --num-workers 2 \
#      --max-sentences $BATCH_SIZE \
#      --no-epoch-checkpoints \
#      --max-sentences-valid $BATCH_SIZE \
#      --lr $LR              \
#      --tokens-per-sample $TOKENS_PER_SAMPLE          \
#      --optimizer adam \
#      --adam-betas '(0.9, 0.95)'  \
#      --adam-eps 10e-8 \
#      --weight-decay 0.1 \
#      --clip-norm $CLIP_NORM      \
#      --max-update $NUM_STEPS     \
#      --total-num-update $NUM_STEPS     \
#      --warmup-updates $NUM_WARMUP_STEPS     \
#      --update-freq $UPDATE_FREQ     \
#      --save-dir ${SERIALIZATION_DIR}/loo_no_sharing_${NUM_GPUS}_GPUs_${ARCH}_${UNTIE_PARAMETERS}_${FILE_SUFFIX}      \
#      --batch-size-valid 2                        \
#      --wandb-project $WANDB_PROJECT           \
#      --valid-subset $valid_subset \
#      --train-domains $domains  \
#      --eval-domains $domains \
#      --required-batch-size-multiple 1 \
#      --memory-efficient-fp16 \
#      --distributed-world-size $NUM_GPUS \
#      --distributed-port $PORT \
#      --desynchronize --domain-parallel \
#      --ddp-backend no_c10d \
#      --sync-type manual \
#      --untie-parameters $UNTIE_PARAMETERS \
#      --data-parallel-groups "${DATA_PARALLEL_GROUPS}" \
#      --all-gather-list-size 32000
#      #--gpu-mappings '{0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1,8:2,9:2,10:2,11:2,12:3,13:3,14:3,15:3,16:3,17:3,18:3,19:3,20:5,21:5,22:5,23:5,24:6,25:6,26:6,27:6:28:7,29:7,30:7,31:7}'
# fi;


# if [[ $EXPERIMENT == *"cluster"* ]]; then
# # None
# srun --label python fairseq_cli/train.py     $DATA_PATH \
#      --task multidomain_language_modeling \
#      --sample-break-mode none \
#      --log-format simple  \
#      --log-interval $LOG_INTERVAL    \
#      --skip-invalid-size-inputs-valid-test     \
#      --validate-interval-updates $VALIDATION_INTERVAL     \
#      --save-interval-updates $SAVE_INTERVAL_UPDATES     \
#      --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
#      --arch $ARCH    \
#      --criterion desynchronized_cross_entropy     \
#      --lr-scheduler polynomial_decay     \
#      --num-workers 2 \
#      --max-sentences $BATCH_SIZE \
#      --no-epoch-checkpoints \
#      --max-sentences-valid $BATCH_SIZE \
#      --lr $LR              \
#      --tokens-per-sample $TOKENS_PER_SAMPLE          \
#      --optimizer adam \
#      --adam-betas '(0.9, 0.95)'  \
#      --adam-eps 10e-8 \
#      --weight-decay 0.1 \
#      --clip-norm $CLIP_NORM      \
#      --max-update $NUM_STEPS     \
#      --total-num-update $NUM_STEPS     \
#      --warmup-updates $NUM_WARMUP_STEPS     \
#      --update-freq $UPDATE_FREQ     \
#      --save-dir ${SERIALIZATION_DIR}/domain_parallel_no_sharing_clustered_${NUM_GPUS}_GPUs_${ARCH}_${UNTIE_PARAMETERS}_${FILE_SUFFIX}      \
#      --batch-size-valid 2                        \
#      --wandb-project $WANDB_PROJECT           \
#      --valid-subset $valid_subset \
#      --train-domains $domains  \
#      --eval-domains $domains \
#      --required-batch-size-multiple 1 \
#      --memory-efficient-fp16 \
#      --distributed-world-size $NUM_GPUS \
#      --distributed-port $PORT \
#      --desynchronize --domain-parallel \
#      --ddp-backend no_c10d \
#      --sync-type manual \
#      --untie-parameters $UNTIE_PARAMETERS \
#      --data-parallel-groups "${DATA_PARALLEL_GROUPS}" \
#      --all-gather-list-size 32000
#      #--gpu-mappings '{0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:1,8:2,9:2,10:2,11:2,12:3,13:3,14:3,15:3,16:3,17:3,18:3,19:3,20:5,21:5,22:5,23:5,24:6,25:6,26:6,27:6:28:7,29:7,30:7,31:7}'
# fi;


# if [[ $EXPERIMENT == *"expert_ffn"* ]]; then
# # None
# srun --label python fairseq_cli/train.py     $DATA_PATH \
#      --task multidomain_language_modeling \
#      --sample-break-mode none \
#      --log-format simple  \
#      --log-interval $LOG_INTERVAL    \
#      --skip-invalid-size-inputs-valid-test     \
#      --validate-interval-updates $VALIDATION_INTERVAL     \
#      --save-interval-updates $SAVE_INTERVAL_UPDATES     \
#      --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
#      --no-epoch-checkpoints \
#      --arch $ARCH    \
#      --criterion desynchronized_cross_entropy     \
#      --lr-scheduler polynomial_decay     \
#      --lr $LR              \
#      --tokens-per-sample $TOKENS_PER_SAMPLE          \
#      --optimizer adam \
#      --adam-betas '(0.9, 0.95)'  \
#      --adam-eps 10e-8 \
#      --weight-decay 0.1 \
#      --num-workers 2 \
#      --max-sentences $BATCH_SIZE \
#      --max-sentences-valid $BATCH_SIZE \
#      --clip-norm $CLIP_NORM      \
#      --max-update $NUM_STEPS     \
#      --total-num-update $NUM_STEPS     \
#      --warmup-updates $NUM_WARMUP_STEPS     \
#      --update-freq $UPDATE_FREQ     \
#      --clip-norm $CLIP_NORM     \
#      --save-dir ${SERIALIZATION_DIR}/domain_parallel_no_sharing_${NUM_GPUS}_GPUs_${ARCH}_${UNTIE_PARAMETERS}_${FILE_SUFFIX}      \
#      --batch-size-valid 2                        \
#      --wandb-project $WANDB_PROJECT           \
#      --valid-subset $valid_subset \
#      --train-domains $domains  \
#      --eval-domains $domains \
#      --required-batch-size-multiple 1 \
#      --memory-efficient-fp16 \
#      --distributed-world-size $NUM_GPUS \
#      --distributed-port $PORT \
#      --desynchronize --domain-parallel \
#      --ddp-backend no_c10d \
#      --sync-type manual \
#      --untie-parameters $UNTIE_PARAMETERS \
#      --data-parallel-groups "${DATA_PARALLEL_GROUPS}"\
#      --add-expert-ffn \
#      --all-gather-list-size 32000
# fi;


# if [[ $EXPERIMENT == *"partial"* ]]; then
# # manual
# srun --label python fairseq_cli/train.py     $DATA_PATH \
#      --task multidomain_language_modeling \
#      --sample-break-mode none \
#      --log-format simple  \
#      --log-interval $LOG_INTERVAL    \
#      --skip-invalid-size-inputs-valid-test     \
#      --validate-interval-updates $VALIDATION_INTERVAL     \
#      --save-interval-updates $SAVE_INTERVAL_UPDATES     \
#      --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
#      --no-epoch-checkpoints \
#      --arch $ARCH    \
#      --criterion desynchronized_cross_entropy     \
#      --lr-scheduler polynomial_decay     \
#      --num-workers 2 \
#      --max-sentences $BATCH_SIZE \
#      --max-sentences-valid $BATCH_SIZE \
#      --lr $LR              \
#      --tokens-per-sample $TOKENS_PER_SAMPLE          \
#      --optimizer adam \
#      --adam-betas '(0.9, 0.95)'  \
#      --adam-eps 10e-8 \
#      --weight-decay 0.1 \
#      --clip-norm $CLIP_NORM      \
#      --max-update $NUM_STEPS     \
#      --total-num-update $NUM_STEPS     \
#      --warmup-updates $NUM_WARMUP_STEPS     \
#      --update-freq $UPDATE_FREQ     \
#      --save-dir ${SERIALIZATION_DIR}/domain_parallel_partial_sharing_${NUM_GPUS}_GPUs_${ARCH}_${UNTIE_PARAMETERS}_${FILE_SUFFIX}      \
#      --batch-size-valid 2                        \
#      --wandb-project $WANDB_PROJECT           \
#      --train-domains $domains  \
#      --eval-domains $domains \
#      --required-batch-size-multiple 1 \
#      --memory-efficient-fp16 \
#      --distributed-world-size $NUM_GPUS \
#      --distributed-port $PORT \
#      --desynchronize --domain-parallel \
#      --ddp-backend no_c10d \
#      --sync-type manual \
#      --untie-parameters $UNTIE_PARAMETERS \
#      --data-parallel-groups "${PARTIAL_SHARING_GROUPS}" \
#      --all-gather-list-size 32000
#      #--data-parallel-groups "0,1,8,9,10,11 2,3 4,5 6,7 12,13 14,15"
#      #--data-parallel-groups "0,1,2,3,16,17,18,19,20,21,22,23 4,5,6,7 8,9,10,11 12,13,14,15 24,25,26,27 28,29,30,31"
# fi

# if [[ $EXPERIMENT == *"random"* ]]; then
# # random
# srun --label python fairseq_cli/train.py     $DATA_PATH \
#      --task multidomain_language_modeling \
#      --sample-break-mode none \
#      --log-format simple  \
#      --log-interval $LOG_INTERVAL    \
#      --skip-invalid-size-inputs-valid-test     \
#      --validate-interval-updates $VALIDATION_INTERVAL     \
#      --save-interval-updates 500     \
#      --keep-interval-updates $KEEP_INTERVAL_UPDATES    \
#      --no-epoch-checkpoints \
#      --arch $ARCH    \
#      --criterion desynchronized_cross_entropy     \
#      --lr-scheduler polynomial_decay     \
#      --lr $LR                \
#      --tokens-per-sample 1024          \
#      --optimizer adam \
#      --adam-betas '(0.9, 0.95)'  \
#      --adam-eps 10e-8 \
#      --weight-decay 0.1 \
#      --num-workers 2 \
#      --max-sentences $BATCH_SIZE \
#      --max-sentences-valid $BATCH_SIZE \
#      --clip-norm $CLIP_NORM      \
#      --max-update $NUM_STEPS     \
#      --total-num-update $NUM_STEPS     \
#      --warmup-updates $NUM_WARMUP_STEPS     \
#      --update-freq $UPDATE_FREQ     \
#      --save-dir ${SERIALIZATION_DIR}/domain_parallel_random_sharing_${NUM_GPUS}_GPUs_large        \
#      --batch-size-valid 2                        \
#      --wandb-project $WANDB_PROJECT           \
#      --train-domains $domains \
#      --eval-domains $domains \
#      --required-batch-size-multiple 1 \
#      --memory-efficient-fp16 \
#      --distributed-world-size $NUM_GPUS \
#      --distributed-port $PORT \
#      --desynchronize --domain-parallel \
#      --ddp-backend no_c10d \
#      --sync-type manual \
#      --untie-parameters $UNTIE_PARAMETERS \
#      --data-parallel-groups "0,5,3 2,7 1 4 6" \
#      --all-gather-list-size 32000
# fi;



