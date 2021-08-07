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

model=$1
direction=$2
type=test

langs="en,cs,de,es,et,fi,fr,gu,hi,kk,lt,lv,ro,ru,tr,zh"
if [ $direction == "en_to_many" ] ; then
    lang_pairs="en-cs,en-de,en-es,en-et,en-fi,en-fr,en-gu,en-hi,en-kk,en-lt,en-lv,en-ro,en-ru,en-tr,en-zh"
    src="en"
else
    lang_pairs="cs-en,de-en,es-en,et-en,fi-en,fr-en,gu-en,hi-en,kk-en,lt-en,lv-en,ro-en,ru-en,tr-en,zh-en"
    tgt="en"
fi

for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
    if [ $direction == "en_to_many" ] ; then
        tgt=$lang
    else
        src=$lang
    fi
    echo "===========STARTING $lang==========="
    datadir="/private/home/shru/wmt_datasets/binarized/$src-$tgt"
    if [ $model == "dense" ] ; then
        basepath="/large_experiments/moe/shru/wmt30/${direction}/.fp16.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.initlr1e-07.warmup4000.lr0.001.clip0.1.drop0.0.wd0.0.ls0.1.seed2.max_pos512.bsz16.no_c10d.det.ves100000000.ngpu32"
    else
        basepath="/large_experiments/moe/shru/wmt30/${direction}_moe_32experts/32experts.fp16.entsrc.SPL_temperature.tmp5.adam.beta0.9_0.98.archtransformer_wmt_en_de.initlr1e-07.warmup4000.lr0.001.clip0.1.drop0.0.wd0.0.ls0.1.seed2.max_pos512.bsz16.2ndexpall.no_c10d.det.ves100000000.ngpu32"
    fi
    mkdir -p ${basepath}/gen
    if [ $model == "dense" ] ; then 
        python fairseq_cli/generate.py \
            $datadir \
            --path ${basepath}/checkpoint_last.pt \
            -s $src -t $tgt \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --gen-subset $type \
            --task translation_multi_simple_epoch \
            --encoder-langtok src --decoder-langtok \
            --langs $langs \
            --lang-pairs ${lang_pairs} \
            --fp16 \
            --pad-to-fixed-length --pad-to-fixed-bsz \
            --batch-size 32 > ${basepath}/gen/${type}.$src-$tgt
    else
        srun python fairseq_cli/generate.py \
            $datadir \
            --path ${basepath}/checkpoint_last.pt \
            -s $src -t $tgt \
            --remove-bpe 'sentencepiece' \
            --beam 5 \
            --gen-subset $type \
            --task translation_multi_simple_epoch \
            --encoder-langtok src --decoder-langtok \
            --langs $langs \
            --lang-pairs ${lang_pairs} \
            --fp16 \
            --distributed-world-size 32 --distributed-port 15187 \
            --pad-to-fixed-length --pad-to-fixed-bsz \
            --batch-size 32 > ${basepath}/gen/${type}.$src-$tgt
    fi
done

