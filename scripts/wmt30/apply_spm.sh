inputdir="/private/home/shru/wmt_datasets/raw"
outputdir="/private/home/shru/wmt_datasets/spm_applied"
logdir="/private/home/shru/wmt_datasets/spm_applied_logs"

mkdir -p ${outputdir}
mkdir -p ${logdir}

# for filepath in `ls ~/wmt_datasets/raw/train.*` ; do
for filename in "$@" ; do
    lp=`echo ${filename} | cut -d'.' -f2`
    echo ${lp}
    src=`echo $lp | cut -d'-' -f1`
    tgt=`echo $lp | cut -d'-' -f2`
    echo $src $tgt
    inputpath="${inputdir}/${filename}"
    outputpath="${outputdir}/${filename}"
    logpath="${logdir}/${filename}"
    echo ${inputpath} ${outputpath}
    python scripts/spm_encode.py \
        --model /private/home/shru/wmt_datasets/spm_64000_400M.model \
        --inputs ${inputpath}.${src} ${inputpath}.${tgt} \
        --outputs ${outputpath}.${src} ${outputpath}.${tgt} > ${logpath} 2>&1
done