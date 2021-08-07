partition=learnfair
num_trials=1
num_nodes=8
num_gpus_per_node=8
experts_per_node=1
num_experts=$(( experts_per_node * num_nodes * num_gpus_per_node ))
type="top2_${num_experts}e"

script_name="sweep_gshard_lm_64experts_64gpus.py"
checkpoint_dir="/checkpoint/$USER/moe_lm/${type}/"
data_dir="/private/home/namangoyal/dataset/data-bin/bookwiki_CC-NEWS_openwebtext_stories_cc100-mmap2-bin"
python fb_sweep/${script_name} -d ${data_dir} -p $type \
    --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
    -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} \
    --resume-failed --time 3999 --mem 470G
