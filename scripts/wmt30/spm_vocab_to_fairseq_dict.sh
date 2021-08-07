SPM_VOCAB="/private/home/shru/wmt_datasets/spm_64000_400M.vocab"
FAIRSEQ_DICT="/private/home/shru/wmt_datasets/spm_64000_400M.dict"
tail -n +4 $SPM_VOCAB | awk '{print $1" "1}' > $FAIRSEQ_DICT