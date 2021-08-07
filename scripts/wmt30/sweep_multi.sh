# bash sweep_multi.sh moe/dense en_to_many/many_to_en
partition=learnfair
num_trials=12
num_nodes=4
num_gpus_per_node=8
type=$2

if [ "$1" == "moe" ] ; then
    script_name="sweep_wmt30_multi_moe.py"
    expert_count=$(( num_nodes * num_gpus_per_node ))
    prefix="${expert_count}experts_v2"
    checkpoint_dir="/checkpoint/$USER/wmt30/${type}_moe_$prefix/"
else
    script_name="sweep_wmt30_multi.py"
    checkpoint_dir="/checkpoint/$USER/wmt30/${type}/"
    prefix="dense"
fi

langs="en,cs,de,es,et,fi,fr,gu,hi,kk,lt,lv,ro,ru,tr,zh"
if [ "$type" == "en_to_many" ] ; then
    lang_pairs="en-cs,en-de,en-es,en-et,en-fi,en-fr,en-gu,en-hi,en-kk,en-lt,en-lv,en-ro,en-ru,en-tr,en-zh"
else
    lang_pairs="cs-en,de-en,es-en,et-en,fi-en,fr-en,gu-en,hi-en,kk-en,lt-en,lv-en,ro-en,ru-en,tr-en,zh-en"
fi

data_dir="/private/home/shru/wmt_datasets/${type}_bin"

python fb_sweep/${script_name} -d ${data_dir} -p "$prefix" \
    --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
    -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
    --time 3999 \
    --sampling-method temperature --sampling-temperature 5 \
    --decoder-langtok --encoder-langtok src \
    --langs $langs \
    --lang-pairs ${lang_pairs} \
    --ddp-backend no_c10d \
    --max-update 200000 \
    --update-freq 4 \
    --save-interval-updates 10000 \
    --virtual-epoch-size 100000000 \
    --dropout 0.0
