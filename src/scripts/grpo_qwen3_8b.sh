#!/bin/bash
export GRPO_REWARD_DEBUG=1
# Distributed training configuration
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}
MASTER_PORT=${MASTER_PORT:-$(shuf -i 20001-29999 -n 1)}
NPROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)  # Automatically detects available GPUs

# DeepSpeed configuration
deepspeed=./scripts/zero2.json

# Model configuration
llm=../checkpoints/qwen3vl8b_5epoch

# Training hyperparameters
lr=1e-6
batch_size=1  
grad_accum_steps=16

# Training entry point
entry_file=qwenvl/train/train_grpo_f360.py

# Dataset configuration
datasets=f360rec

# Output configuration
run_name="qwen3vl8b_5epoch_grpo_b1_16_g8_f360_2epochs"
output_dir=../checkpoints/${run_name}

# Resume (optional)
# Override by exporting RESUME_FROM_CHECKPOINT=/path/to/checkpoint-xxx
# resume_from_checkpoint=${RESUME_FROM_CHECKPOINT:-"/path/to/checkpoint-xxx"}

# Training arguments
args="
    --deepspeed ${deepspeed} \
    --model_name_or_path "${llm}" \
    --dataset_use ${datasets} \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --tune_mm_llm True \
    --bf16 \
    --output_dir ${output_dir} \
    --num_train_epochs 2 \
    --per_device_train_batch_size ${batch_size} \
    --gradient_accumulation_steps ${grad_accum_steps} \
    --max_pixels 50176 \
    --min_pixels 784 \
    --eval_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --learning_rate ${lr} \
    --weight_decay 0 \
    --warmup_ratio 0.03 \
    --max_grad_norm 1 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --model_max_length 8192 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --run_name ${run_name} \
    --report_to tensorboard"

if [[ -n "${resume_from_checkpoint}" ]]; then
    args+=" --resume_from_checkpoint ${resume_from_checkpoint}"
fi

# Launch training
torchrun --nproc_per_node=${NPROC_PER_NODE} \
         --master_addr=${MASTER_ADDR} \
         --master_port=${MASTER_PORT} \
         ${entry_file} ${args}