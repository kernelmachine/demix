NUM_GPUS=$1
NUM_NODES=$((${NUM_GPUS}/8))
PORT=$2
UNTIE_PARAMETERS=$3
ARCH=$4
DATA_DIR=$5
DOMAIN=$6

echo "requesting ${NUM_GPUS} GPUs over ${NUM_NODES} nodes..."
echo "salloc --gpus-per-node 8 --nodes ${NUM_NODES}  --ntasks-per-node 8 --cpus-per-task 10 --mem 480G --time 3000 --partition learnfair,dev"

#salloc --gpus-per-node 8 --nodes $NUM_NODES  --ntasks-per-node 8 --cpus-per-task 10 --mem 480G --time 3000 --partition learnfair,dev

# None
srun --label python fairseq_cli/train.py     $DATA_DIR \
     --task multidomain_language_modeling \
     --sample-break-mode none \
     --log-format simple  \
     --log-interval 50     \
     --skip-invalid-size-inputs-valid-test     \
     --validate-interval-updates 5000     \
     --save-interval-updates 500     \
     --keep-interval-updates 1    \
     --arch $ARCH    \
     --criterion cross_entropy     \
     --lr-scheduler inverse_sqrt     \
     --warmup-init-lr 1e-07     \
     --lr 3e-4               \
     --tokens-per-sample 512          \
     --weight-decay 0.01           \
     --dropout 0.1  \
     --optimizer adam \
     --adam-betas '(0.9, 0.98)'  \
     --clip-norm 0.0      \
     --max-update 500     \
     --warmup-updates 1000     \
     --update-freq 8     \
     --clip-norm 0.0     \
     --save-dir final_results/domain_parallel_partial_sharing_${NUM_GPUS}_GPUs_large_${DOMAIN}_multidapt        \
     --batch-size-valid 2                        \
     --train-domains 1b,cs,legal,biomed,webtext,realnews,tweets,reviews \
     --eval-domains 1b,cs,legal,biomed,webtext,realnews,tweets,reviews \
     --disable-validation \
     --batch-size 16 \
     --ddp-backend no_c10d \
     --desynchronize \
     --distributed-world-size $NUM_GPUS \
     --distributed-port $PORT \
     --untie-parameters feedforward \
     --sync-type manual --fp16 --fp16-no-flatten-grads \
     --data-parallel-groups "0,4,7 1 2 3 6 5" \
     --finetune-from-model final_results/data_parallel_8_GPUs_large/checkpoint_last.pt
     #--data-parallel-groups "0,1 2,3 4,5 6,7 8,9 10,11 12,13 14,15"
     #--data-parallel-groups "0,1,2,3 4,5,6,7 8,9,10,11 12,13,14,15 16,17,18,19 20,21,22,23 24,25,26,27 28,29,30,31"

