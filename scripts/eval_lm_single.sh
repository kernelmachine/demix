data_bin=$1
model=$2
results_path=$3
split=$4
target_domain=$5
force_domain_token=$6

if [[ "$model" == *"domain_token"* ]]; then
        if [[ -z "$force_domain_token" ]]; then
                python fairseq_cli/eval_lm.py \
                        ${data_bin} \
                        --path ${model} \
                        --gen-subset ${split}_${target_domain} \
                        --task multidomain_language_modeling \
                        --sample-break-mode none \
                        --tokens-per-sample 1024     \
                        --batch-size 2  \
                        --original-domains 1b,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit \
                        --eval-domains ${target_domain} \
                        --results-path ${results_path} \
                        --partial-load \
                        --add-domain-token;
        else 
                python fairseq_cli/eval_lm.py \
                        ${data_bin} \
                        --path ${model} \
                        --gen-subset ${split}_${target_domain} \
                        --task multidomain_language_modeling \
                        --sample-break-mode none \
                        --tokens-per-sample 1024     \
                        --batch-size 2  \
                        --original-domains 1b,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit \
                        --eval-domains ${target_domain} \
                        --results-path ${results_path} \
                        --partial-load \
                        --add-domain-token \
                        --force-domain-token $force_domain_token;
        fi;
elif [[ "$model" == *"gshard"* || "$model" == *"switch"* ]]; then
  srun --label python fairseq_cli/eval_lm.py \
        ${data_bin} \
        --path ${model} \
        --gen-subset ${split}_${target_domain} \
        --task multidomain_language_modeling \
        --sample-break-mode none \
        --tokens-per-sample 1024     \
        --batch-size 2  \
        --original-domains 1b,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit \
        --eval-domains ${target_domain} \
        --results-path ${results_path} \
        --distributed-world-size 64 \
        --distributed-port 4234 \
	--is-moe;
else
        python fairseq_cli/eval_lm.py \
        ${data_bin} \
        --path ${model} \
        --gen-subset ${split}_${target_domain} \
        --task multidomain_language_modeling \
        --sample-break-mode none \
        --tokens-per-sample 1024     \
        --batch-size 2  \
        --original-domains 1b,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit \
        --eval-domains ${target_domain} \
        --results-path ${results_path} \
	--partial-load
fi





# python fairseq_cli/eval_lm.py \
#         ${data_bin} \
#         --path ${model} \
#         --gen-subset valid_1b,valid_biomed,valid_cs,valid_legal,valid_realnews,valid_reviews,valid_tweets,valid_webtext\
#         --task multidomain_language_modeling \
#         --sample-break-mode none \
#         --tokens-per-sample 1024     \
#         --batch-size 2  \
#         --original-domains 1b,biomed,cs,legal,realnews,reviews,tweets,webtext \
#         --eval-domains 1b,biomed,cs,legal,realnews,reviews,tweets,webtext \
#         --results-path ${result_path} \
#         #--add-bos-token

