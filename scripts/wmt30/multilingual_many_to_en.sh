nodes=1
gpus=8
ws=$(( nodes * gpus ))

data="/private/home/shru/wmt_datasets/many_to_en_bin"
#langs="en,cs,de,es,et,fi,fr,gu,hi,kk,lt,lv,ro,ru,tr,zh"
#lang_pairs="cs-en,de-en,es-en,et-en,fi-en,fr-en,gu-en,hi-en,kk-en,lt-en,lv-en,ro-en,ru-en,tr-en,zh-en"
# data="/private/home/shru/wmt_datasets/binarized/et-en"
langs="en,et,fi"
lang_pairs="et-en,fi-en"
save_dir="chk_demo"
python train.py $data \
--task translation_multi_simple_epoch \
--decoder-langtok --encoder-langtok src \
--langs $langs \
--lang-pairs  $lang_pairs \
--sampling-method=temperature \
--sampling-temperature=5 \
--distributed-world-size $ws \
--distributed-port 11518 \
--save-dir $save_dir \
--fp16 --fp16-no-flatten-grads \
--ddp-backend no_c10d \
--max-update 200000 \
--arch transformer_wmt_en_de_big \
--optimizer adam --adam-betas '(0.9, 0.98)' \
--lr-scheduler inverse_sqrt --warmup-init-lr 1e-07 --warmup-updates 4000 --lr 0.001 --min-lr 1e-09 --clip-norm 0.1 \
--dropout 0.0 --weight-decay 0.0 \
--criterion label_smoothed_cross_entropy --label-smoothing 0.1 \
--seed 2 --log-format json --log-interval 1 --save-interval-updates 10000 \
--validate-interval-updates 2000 \
--max-source-positions 256 --max-target-positions 256 \
--batch-size 16 \
--pad-to-fixed-length \
--pad-to-fixed-bsz \
--batch-size-valid 2 \
--train-subset valid


# --criterion moe_label_smoothed_cross_entropy --label-smoothing 0.1 \
# --moe-gate-loss-wt 0.01 \
# --moe-gate-loss-combine-method sum \
# --moe-second-expert-policy all \
# --moe-gating-use-fp32 \
# --moe-freq 2 \
# --moe-expert-count $ws \


# --max-tokens 3584 \

# --task translation -s et -t en \
