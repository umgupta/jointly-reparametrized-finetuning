#!/usr/bin/env bash

USE_MASK_EMBEDDINGS=$5
USE_DROPOUT=False
MODEL_KEY="jr-warp-tuning"
MODEL=$8
TASK=$7
z_size=$5
z_init=$6
w_init=$6
ROOT_DIR="./"

LR=$3
num_epochs=20
noise=$1
per_sample_grad_norm=$2
gradient_accumulation_steps=$4

python -m src.run_private_glue \
  --model_name_or_path "$MODEL" \
  --task_name "$TASK" \
  --do_train \
  --do_eval \
  --dataloader_drop_last True \
  --pad_to_max_length False \
  --per_device_train_batch_size 16 \
  --gradient_accumulation_steps "$gradient_accumulation_steps" \
  --save_strategy "no" \
  --seed 0 \
  --save_total_limit 1 \
  --logging_steps 1 \
  --warmup_ratio 0.0 \
  --learning_rate "$LR" \
  --max_grad_norm -1 \
  --lr_scheduler_type constant \
  --num_train_epochs "$num_epochs" \
  --output_dir $ROOT_DIR/results/private/"$TASK"/"$MODEL"/"$MODEL_KEY"/noise_"$noise"_norm_"$per_sample_grad_norm"/z_size_"$z_size"/lr_"$LR"_epoch_"$num_epochs"_use_mask_"$USE_MASK_EMBEDDINGS"_use_dropout_"$USE_DROPOUT"_z_init_"$z_init"_w_init_"$w_init" \
  --overwrite_output_dir False \
  --model_key $MODEL_KEY \
  --use_mask_embeddings "$USE_MASK_EMBEDDINGS" \
  --use_dropout $USE_DROPOUT \
  --z_size "$z_size" --z_init "$z_init" --w_init "$w_init" \
  --noise "$noise" --per_sample_grad_norm "$per_sample_grad_norm"
