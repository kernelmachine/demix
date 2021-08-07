partition=moe,learnfair
num_trials=1
num_nodes=4
num_gpus_per_node=8

type=many_to_en

script_name="sweep_wmt30_multi.py"
checkpoint_dir="/large_experiments/moe/shru/wmt30/${type}/"
data_dir="/private/home/shru/wmt_datasets/${type}_bin"
python fb_sweep/${script_name} -d ${data_dir} -p "" \
    --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
    -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed \
    --time 3999 \
    --sampling-method temperature --sampling-temperature 5 \
    --decoder-langtok --encoder-langtok src \
    --langs en,cs,de,es,et,fi,fr,gu,hi,kk,lt,lv,ro,ru,tr,zh \
    --lang-pairs cs-en,de-en,es-en,et-en,fi-en,fr-en,gu-en,hi-en,kk-en,lt-en,lv-en,ro-en,ru-en,tr-en,zh-en \
    --ddp-backend no_c10d \
    --max-update 200000 \
    --update-freq 1 \
    --save-interval-updates 10000 \
    --virtual-epoch-size 100000000 \
    --dropout 0.0
    
    