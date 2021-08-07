/private/home/namangoyal/src/sentencepiece/build/src/spm_train \
--input "/private/home/shru/wmt_datasets/raw/spm_samples.txt" \
--model_prefix "/private/home/shru/wmt_datasets/spm_64000_400M" \
--vocab_size=64000 \
--character_coverage=0.99995 \
--model_type=bpe \
--shuffle_input_sentence=true \
--input_sentence_size=400000000 \
--num_threads 72;