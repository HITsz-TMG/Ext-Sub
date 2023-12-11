#!/bin/bash

model_name_or_path=huggyllama/llama-7b
data_path=
output_dir=

torchrun --nproc_per_node=2 --master_port=1234 train_peft.py \
    --model_name_or_path ${model_name_or_path} \
    --data_path ${data_path} \
    --bf16 True \
    --output_dir ${output_dir} \
    --overwrite_output_dir \
    --num_train_epochs 2 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 16 \
    --model_max_length 1024 \
    --evaluation_strategy "no" \
    --logging_steps 1 \
    --save_strategy "steps" \
    --save_steps 4000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --deepspeed "./configs/zero2_offload_opt_param.json" \
    --tf32 True

