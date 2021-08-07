data_path=$1
model=$2
output_dir=$3

if [ ! -d "$output_dir" ]; then
  mkdir -p "$output_dir"
fi

bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-0.pt ${output_dir}/rank_0.jsonl
bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-1.pt ${output_dir}/rank_1.jsonl
bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-2.pt ${output_dir}/rank_2.jsonl
bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-3.pt ${output_dir}/rank_3.jsonl
bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-4.pt ${output_dir}/rank_4.jsonl
bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-5.pt ${output_dir}/rank_5.jsonl
bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-6.pt ${output_dir}/rank_6.jsonl
bash scripts/eval_desync_lm.sh ${data_path} ${model}/checkpoint_last-rank-7.pt ${output_dir}/rank_7.jsonl
