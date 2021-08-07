data_bin=$1
model=$2
results_path=$3
split=$4
zero_shot=$5

INDEX=0
echo $domains

#for tg in "gutenberg",2 "github_redo",1 "anonymized_tweets_redo",7 "cord19-redo",6 "anonymized_latest_news_redo",1 "anonymized_yelp_reviews_redo",3 "legal_contracts",1 "qasper",4; do
for tg in "1b",0 "anonymized_openwebtext",1 "anonymized_realnews",2 "anonymized_reviews",3 "cs",4 "legal",5 "med",6 "reddit",7; do
#for tg in "1b",0 "cs",1 "legal",2 "med",3 "anonymized_openwebtext",4 "anonymized_realnews",5 "reddit",6 "anonymized_reviews",7; do
        IFS=',' read target_domain INDEX <<< "${tg}"
        let RANK=${INDEX}*4
        python fairseq_cli/eval_lm.py \
        ${data_bin} \
        --path ${model}/checkpoint_last-rank-${RANK}.pt \
        --gen-subset ${split}_${target_domain} \
        --task multidomain_language_modeling \
        --sample-break-mode none \
        --tokens-per-sample 1024     \
        --batch-size 2  \
        --original-domains gutenberg,anonymized_openwebtext,anonymized_realnews,anonymized_reviews,cs,legal,med,reddit \
        --eval-domains ${target_domain} \
        --results-path ${results_path} \
        --partial-load
done;

