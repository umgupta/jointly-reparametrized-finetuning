#!/usr/bin/env bash

echo "Running SLaSh"
python -m src.run_time \
  --model_name_or_path "roberta-base" \
  --task_name "qnli" \
  --do_train True\
  --do_eval False \
  --pad_to_max_length False \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --num_train_epochs 1 \
  --overwrite_output_dir True \
  --model_key "slash-tuning" \
  --use_mask_embeddings False \
  --use_dropout False \
  --z_size 5000 \
  --output_dir tmp/ 2>&1 > layer_shift.test.log

echo "Running Full Tuning"
python -m src.run_time \
  --model_name_or_path "roberta-base" \
  --task_name "qnli" \
  --do_train True\
  --do_eval False \
  --pad_to_max_length False \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --num_train_epochs 1 \
  --overwrite_output_dir True \
  --model_key "full-finetune" \
  --use_mask_embeddings False \
  --use_dropout False \
  --output_dir tmp/ 2>&1 > full_finetune.log


echo "Running BitFit"
python -m src.run_time \
  --model_name_or_path "roberta-base" \
  --task_name "qnli" \
  --do_train True\
  --do_eval False \
  --pad_to_max_length False \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --num_train_epochs 1 \
  --overwrite_output_dir True \
  --model_key "bitfit" \
  --use_mask_embeddings False \
  --use_dropout False \
  --output_dir tmp/ 2>&1 > bitfit.test.log

echo "Running LoRA"
python -m src.run_time \
  --model_name_or_path "roberta-base" \
  --task_name "qnli" \
  --do_train True\
  --do_eval False \
  --pad_to_max_length False \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --num_train_epochs 1 \
  --overwrite_output_dir True \
  --model_key "lora" \
  --use_mask_embeddings False \
  --use_dropout False \
  --lora_rank 1 \
  --output_dir tmp/ 2>&1 > lora_rank_1.test.log

echo "Running Adapters"
python -m src.run_time \
  --model_name_or_path "roberta-base" \
  --task_name "qnli" \
  --do_train True\
  --do_eval False \
  --pad_to_max_length False \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --num_train_epochs 1 \
  --overwrite_output_dir True \
  --model_key "adapters" \
  --use_mask_embeddings False \
  --use_dropout False \
  --adapters_reduction_factor 768 \
  --output_dir tmp/ 2>&1 > adapters_rf_768.test.log

echo "Running WARP"
python -m src.run_time \
  --model_name_or_path "roberta-base" \
  --task_name "qnli" \
  --do_train True\
  --do_eval False \
  --pad_to_max_length False \
  --per_device_train_batch_size 8 \
  --gradient_accumulation_steps 1 \
  --save_strategy "no" \
  --seed 0 \
  --num_train_epochs 1 \
  --overwrite_output_dir True \
  --model_key "warp" \
  --use_mask_embeddings False \
  --use_dropout False \
  --warp_len 20 \
  --output_dir tmp/ 2>&1 > warp_len_20.test.log

