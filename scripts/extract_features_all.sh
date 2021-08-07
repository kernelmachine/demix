data_path=$1

domains=$(find $data_path -maxdepth 1 -type d -exec basename {} \;)

for domain in $domains; do \
    if [ -f "$data_path/${domain}/${domain}.train.jsonl"  ]; then
        echo "extracting features from ${domain}"
        python scripts/extract_features.py --model roberta-base \
                                    --output_dir ${data_path}/${domain}/${domain}_embeddings \
                                    --input_file ${data_path}/${domain}/${domain}.train.jsonl \
                                    --batch_size 16  \
                                    --num_shards 10;
        python scripts/convert_pytorch_to_memmap.py "${data_path}/${domain}/${domain}_embeddings/*"
    else
        echo "${data_path}/${domain}/${domain}.train.jsonl not found"
    fi
done
