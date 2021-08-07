#!/bin/bash
# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


# raw glue data as downloaded by glue download script (https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e)
if [[ $# -ne 3 ]]; then
  echo "Run as following:"
  echo "./examples/few_shot/pet/scripts/preprocess_fewglue.sh <data_folder> <task_name> <pattern_id>"
  exit 1
fi

DATA_FOLDER=$1

# download bpe encoder.json, vocabulary and fairseq dictionary
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe'
wget -N 'https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/dict.txt'

TASKS=$2 # BoolQ
PATTERN_ID=$3

if [ "$TASKS" = "ALL" ]
then
  TASKS="BoolQ"
fi

for TASK in $TASKS
do
  echo "Preprocessing $TASK"

  TASK_DATA_FOLDER="$DATA_FOLDER/$TASK"
  if [ "$PATTERN_ID" == "-1" ]
  then
    TASK_OUTPUT_FOLDER="$TASK_DATA_FOLDER/baseline"
  else
    TASK_OUTPUT_FOLDER="$TASK_DATA_FOLDER/$PATTERN_ID"
  fi

  rm -rf $TASK_OUTPUT_FOLDER
  mkdir -p "$TASK_OUTPUT_FOLDER/processed"

  python ./examples/few_shot/pet/scripts/preprocess_fewglue.py \
    --encoder-json encoder.json \
    --vocab-bpe vocab.bpe \
    --input-dir $TASK_DATA_FOLDER \
    --output-dir "$TASK_OUTPUT_FOLDER/processed" \
    --task $TASK \
    --pattern-id  $PATTERN_ID
 
  mkdir -p "$TASK_OUTPUT_FOLDER/bin"
  # Run fairseq preprocessing:
  fairseq-preprocess \
    --only-source \
    --trainpref "$TASK_OUTPUT_FOLDER/processed/train.input0.bpe" \
    --validpref "$TASK_OUTPUT_FOLDER/processed/valid.input0.bpe" \
    --destdir "$TASK_OUTPUT_FOLDER/bin/input0" \
    --workers 60 \
    --task multiple_choice_mlm \
    --srcdict dict.txt;

  fairseq-preprocess \
    --only-source \
    --trainpref "$TASK_OUTPUT_FOLDER/processed/train.label.bpe" \
    --validpref "$TASK_OUTPUT_FOLDER/processed/valid.label.bpe" \
    --destdir "$TASK_OUTPUT_FOLDER/bin/label" \
    --workers 60
done
