#lr=0.001

declare -A test_set
test_set=( ["de-en"]="wmt14" ["ro-en"]="wmt16" ["cs-en"]="wmt18" \
["fr-en"]="wmt14" ["ru-en"]="wmt19" ["zh-en"]="wmt19" ["es-en"]="wmt13" 
["fi-en"]="wmt19" ["et-en"]="wmt18" ["lv-en"]="wmt17" ["lt-en"]="wmt19" 
["hi-en"]="wmt14" ["kk-en"]="wmt19" ["tr-en"]="wmt18" ["gu-en"]="wmt19" )

declare -A valid_set
valid_set=( ["cs-en"]="wmt17" ["fr-en"]="wmt13" ["ru-en"]="wmt18" \
["zh-en"]="wmt18" ["es-en"]="wmt12" ["fi-en"]="wmt18" ["de-en"]="wmt13" \
["et-en"]="wmt18/dev" ["lv-en"]="wmt17/dev" ["lt-en"]="wmt19/dev" ["ro-en"]="wmt16/dev" \
["hi-en"]="wmt14" ["kk-en"]="wmt19/dev" ["tr-en"]="wmt17" ["gu-en"]="wmt19/dev" )

for type in valid test ; do
    echo "======" $type "===="
# for lang_pair in cs-en fr-en ru-en zh-en es-en fi-en et-en lv-en lt-en hi-en kk-en tr-en gu-en ; do
for basepath in `ls -d /large_experiments/moe/shru/wmt30/bilingual_base/*-en/*/` ; do
    lang_pair=`echo $basepath | cut -d'/' -f7`
    src=`echo ${lang_pair} | cut -d'-' -f1`
    tgt=`echo ${lang_pair} | cut -d'-' -f2`
    #basepath="/large_experiments/moe/shru/wmt30/bilingual_base/${lang_pair}/${lang_pair}.fp16.fp16_no_flatten.c10d.archtransformer_wmt_en_de.adam.beta0.9_0.98.initlr1e-07.warmup4000.lr${lr}.clip0.1.drop0.1.wd0.0.ls0.1.maxtok3584.seed2.ngpu16"
    path="${basepath}/checkpoint_best.pt"
    if [ -d /private/home/shru/wmt_datasets/binarized/${lang_pair} ] ; then
        datadir=/private/home/shru/wmt_datasets/binarized/${lang_pair}
    else
        datadir=/private/home/shru/wmt_datasets/binarized/$tgt-$src
    fi
    #if [ ! -f ${basepath}/${type}.gen.$src-$tgt ] ; then
        echo "writing generations to ${basepath}/${type}.gen.$src-$tgt"
        python fairseq_cli/generate.py \
            $datadir \
            --path ${path} \
            -s $src -t $tgt \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --gen-subset $type \
            --fp16 > ${basepath}/${type}.gen.$src-$tgt
    # else
    #     echo `wc -l ${basepath}/${type}.gen.$src-$tgt`
    # fi
    # TODO: calculate bleu and store somewhere
    echo $lang_pair
    key_lang_pair=$lang_pair
    if [ $src == "en" ] ; then
        key_lang_pair="$tgt-en"
    fi
    if [ "${type}" == "valid" ] ; then
        testset="${valid_set[${key_lang_pair}]}"
    else
        testset="${test_set[${key_lang_pair}]}"
    fi
    if [ "$lang_pair" == "hi-en" ] || [ "$lang_pair" == "en-hi" ] ; then
        cat ${basepath}/${type}.gen.$src-$tgt | grep -P "^H" | sort -V | cut -f 3- | sacrebleu ~/wmt_datasets/raw/${type}.hi-en.$tgt > ${basepath}/${type}.bleu.$src-$tgt
    else
        cat ${basepath}/${type}.gen.$src-$tgt | grep -P "^H" | sort -V | cut -f 3- | sacrebleu -t $testset -l ${lang_pair} > ${basepath}/${type}.bleu.$src-$tgt
    fi

done
done