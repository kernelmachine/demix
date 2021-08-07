data_bin=$1
model=$2
results_path=$3
# target_domain=$4

        
 python fairseq_cli/eval_lm.py \
         ${data_bin} \
         --path ${model} \
         --gen-subset valid_1b,valid_cs,valid_legal,valid_med,valid_openwebtext,valid_realnews,valid_reddit,valid_reviews \
         --task multidomain_language_modeling \
         --sample-break-mode none \
         --tokens-per-sample 512     \
         --batch-size 2  \
         --original-domains 1b,cs,legal,med,openwebtext,realnews,reddit,reviews \
         --eval-domains 1b,cs,legal,med,openwebtext,realnews,reddit,reviews \
         --partial-load \
         --results-path ${results_path}

#  python fairseq_cli/eval_lm.py \
#          ${data_bin} \
#          --path ${model} \
#          --gen-subset valid_ag,valid_securitystackexchange,valid_imdb,valid_ark_legal,valid_articles,valid_reddit \
#          --task multidomain_language_modeling \
#          --sample-break-mode none \
#          --tokens-per-sample 512     \
#          --batch-size 2  \
#          --original-domains 1b,biomed,cs,legal,realnews,reviews,tweets,webtext \
#          --eval-domains ag,securitystackexchange,imdb,ark_legal,articles,reddit \
#          --partial-load \
#          --results-path ${results_path}
