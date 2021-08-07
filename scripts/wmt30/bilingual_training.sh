# lang_pair="es-en"
# cs-en  de-en  es-en  et-en  fi-en  fr-en  gu-en  hi-en  kk-en  lt-en  lv-en  ro-en  ru-en  tr-en  zh-en
# for lang_pair in cs-en et-en fi-en gu-en lt-en lv-en ro-en ru-en tr-en zh-en ; do
for lang in cs de es  et  'fi'  fr  gu  hi  kk  lt  lv  ro  ru  tr  zh ; do
# for lang in lt es ; do
# for lang in lt 'fi' ; do
    lang_pair="en-${lang}"
    script_name="sweep_wmt30.py"
    checkpoint_dir="/large_experiments/moe/shru/wmt30/bilingual_base/${lang_pair}/"
    data_dir="/private/home/shru/wmt_datasets/binarized/${lang_pair}"

    partition=moe,learnfair
    num_trials=2
    num_nodes=2
    num_gpus_per_node=8

    python fb_sweep/${script_name} -d ${data_dir} -p ${lang_pair} --pair ${lang_pair} \
        --no-tensorboard --checkpoints-dir ${checkpoint_dir} --partition ${partition} --constraint volta32gb \
        -t ${num_trials} -n ${num_nodes} -g ${num_gpus_per_node} --resume-failed
done