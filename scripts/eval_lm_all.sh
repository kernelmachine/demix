data_bin=$1
model=$2
results_path=$3
split=$4
zero_shot=$5

if [[ $zero_shot == "zero_shot" ]]; then
        for target_domain in gutenberg github_redo anonymized_tweets_redo cord19_redo anonymized_latest_news_redo anonymized_yelp_reviews_redo qasper legal_contracts; do
                if [[ "$model" == *"domain_token"* ]]; then
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
                                --partial-load
                fi;
        done;
else

        for target_domain in 1b cs legal med anonymized_openwebtext anonymized_realnews reddit anonymized_reviews; do
                if [[ "$model" == *"domain_token"* ]]; then
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
                        --partial-load
                fi;
        done;
fi;
