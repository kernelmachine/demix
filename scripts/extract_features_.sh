#!/bin/bash
#SBATCH --job-name=mode
#SBATCH --partition=scavenge
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=25G
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00

python -m scripts.extract_features --model $model --output_file $output_file --input_file $input_file --batch_size 8 --device 0
