inputdir="/private/home/shru/wmt_datasets/spm_applied"
outputdir="/private/home/shru/wmt_datasets/binarized"
dict="/private/home/shru/wmt_datasets/spm_64000_400M.dict"

mkdir -p ${outputdir}
mkdir -p ${logdir}

for lp in "$@" ; do
    echo $lp
    src=`echo ${lp} | cut -d'-' -f1`
    tgt=`echo ${lp} | cut -d'-' -f2`
    if [ $src == "en" ] ; then
        python fairseq_cli/preprocess.py \
            -s ${src} -t ${tgt} \
            --validpref ${inputdir}/valid.${lp} \
            --testpref ${inputdir}/test.${lp} \
            --destdir ${outputdir}/${lp} \
            --srcdict ${dict} \
            --joined-dictionary \
            --workers 72
            #--trainpref ${inputdir}/train.${lp} \
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.$tgt.bin ${outputdir}/en-$tgt/train.en-$tgt.$tgt.bin
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.$tgt.idx ${outputdir}/en-$tgt/train.en-$tgt.$tgt.idx
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.en.bin ${outputdir}/en-$tgt/train.en-$tgt.en.bin
        ln -s ${outputdir}/$tgt-en/train.$tgt-en.en.idx ${outputdir}/en-$tgt/train.en-$tgt.en.idx
    else
        python fairseq_cli/preprocess.py \
            -s ${src} -t ${tgt} \
            --trainpref ${inputdir}/train.${lp} \
            --validpref ${inputdir}/valid.${lp} \
            --testpref ${inputdir}/test.${lp} \
            --destdir ${outputdir}/${lp} \
            --srcdict ${dict} \
            --joined-dictionary \
            --workers 72
    fi

done