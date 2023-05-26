#!/usr/bin/env bash

USE_MASK_EMBEDDINGS=True
USE_DROPOUT=False
MODEL_KEY="slash-tuning"
MODEL=$7
TASK=$6
z_size=$5
z_init=$1
w_init=$2
ROOT_DIR="./"

LR=$3
num_epochs=$4
python -m src.run_glue \
  --model_name_or_path "$MODEL" \
  --task_name "$TASK" \
  --do_train \
  --do_eval \
  --pad_to_max_length False \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --warmup_ratio 0.06 \
  --learning_rate "$LR" \
  --num_train_epochs "$num_epochs" \
  --overwrite_output_dir False \
  --model_key $MODEL_KEY \
  --use_mask_embeddings $USE_MASK_EMBEDDINGS \
  --use_dropout $USE_DROPOUT \
  --z_size "$z_size" --z_init "$z_init" --w_init "$w_init" \
  --output_dir "$ROOT_DIR"/results/"$TASK"/"$MODEL"/"$MODEL_KEY"/z_size_"$z_size"/lr_"$LR"_epoch_"$num_epochs"_use_mask_"$USE_MASK_EMBEDDINGS"_use_dropout_"$USE_DROPOUT"_z_init_"$z_init"_w_init_"$w_init"

