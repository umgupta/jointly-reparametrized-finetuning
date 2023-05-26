#!/usr/bin/env bash

USE_DROPOUT=True
MODEL_KEY="linear-classifier"
MODEL=$1
ROOT_DIR="./"
LR=$2
num_epochs=$3
batch_size=$4

python3 -m src.run_ner \
  --model_name_or_path "$MODEL" \
  --dataset_name conll2003 \
  --output_dir "$ROOT_DIR"/ner_results/"$MODEL"/linear_classifier/lr_"$LR"_epochs_"$num_epochs"_batch_size_"$batch_size"\
  --do_train \
  --do_eval \
  --do_predict \
  --pad_to_max_length False \
  --per_device_train_batch_size "$batch_size" \
  --per_device_eval_batch_size 32 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --warmup_ratio 0.1 \
  --learning_rate "$LR" \
  --num_train_epochs "$num_epochs" \
  --overwrite_output_dir False \
  --model_key $MODEL_KEY \
  --use_dropout $USE_DROPOUT \
  --return_entity_level_metrics True
