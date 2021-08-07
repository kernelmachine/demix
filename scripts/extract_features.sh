input_file=$1
shards_dir=$2
num_shards=$3
model=$4
num_lines=$(cat ${input_file} | wc -l)
num_lines_per_shard=`expr ${num_lines} / ${num_shards}`
echo $num_lines_per_shard
mkdir -p $shards_dir
split --lines $num_lines_per_shard --numeric-suffixes $input_file $shards_dir

for f in $shards_dir/*;
    do 
    sbatch --export=ALL,model=$model,input_file=$f,output_file=${f}.pt scripts/extract_features_.sh;
    done
