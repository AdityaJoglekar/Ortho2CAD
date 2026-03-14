#!/bin/bash
source ../../miniconda3/bin/activate

export HF_HOME="../../cache/huggingface"
echo "Running test"


### SFT
# Train 
conda activate vlmtrl
./scripts/sft_qwen3_8b.sh
# Inference
bash ./scripts/test_gencadcode.sh "../checkpoints/qwen3vl8b_5epoch" "cadquery_test_data_subset100"
# Generate CAD and compute IoU
conda activate cad_iou
python3 ./scripts/generate_model_cad.py --dataset_name cadquery_test_data_subset100 --model_tested qwen3vl8b_5epoch --code_language cadquery --parallel
python3 ./scripts/compute_iou.py --model_path qwen3vl8b_5epoch --test_set_name cadquery_test_data_subset100


### RL
# Train
conda activate vlmtrl
./scripts/grpo_qwen3_8b.sh
# Inference
bash ./scripts/test_gencadcode.sh "../checkpoints/qwen3vl8b_5epoch_grpo_b1_16_g8_onlyRL_f360_2epochs/checkpoint-1724" "f360rec_test_data_subset100" "../inference/test100_images_f360" 
# Generate CAD and compute IoU
conda activate cad_iou
python3 ./scripts/generate_model_cad.py --dataset_name f360rec_test_data_subset100 --model_tested checkpoint-1724 --code_language cadquery --parallel
python3 ./scripts/compute_iou.py --model_path checkpoint-1724 --test_set_name f360rec_test_data_subset100 --ground_truth_steps_dir ../inference/test100_gt_steps_f360/


echo "Done"