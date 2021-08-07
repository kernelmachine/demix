SPM_SAMPLE_LINE_COUNT_FILE="/private/home/shru/wmt_datasets/raw/files_to_resampled_lines.tsv"
SPM_SAMPLE_FILE="/private/home/shru/wmt_datasets/raw/spm_samples.txt"

while IFS=$'\t' read -r lang size file; do
  echo "$lang $size $file"
  nl=`cat $file |wc -l`

  while [ $size -ge $nl ]; do
    echo "$lang $size > $nl, dump entire file"
    cat $file >> ${SPM_SAMPLE_FILE}
    size=`echo "$size - $nl"| bc`
  done
  echo "$lang $nl dump remain $size"
  cat $file | shuf | head -n $size >> ${SPM_SAMPLE_FILE}
done < ${SPM_SAMPLE_LINE_COUNT_FILE}