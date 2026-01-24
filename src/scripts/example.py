export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_coco_base65cate_6k.txt"

# export LD_LIBRARY_PATH=/opt/conda/envs/grpo/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda

export DATA_PATH=/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/test500_v2
export CKPT_PATH=/work/docker/GRPO/grpo-lumina/share_models/RealSR-R1_lr5e6/checkpoint-30 #/work/docker/GRPO/PURE-main/pure/ckpts/Alpha-VLLM/PRUE #/work//docker/GRPO/grpo-lumina/Qwen2-VL-2B-Instruct # 
export SAVE_PATH=./share_models/RealSR-R1_lr2e6

export PYTHONPATH=$PYTHONPATH:/work//docker/GRPO/grpo-lumina/src/virft/src
 #8192
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6
export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="11317" \
    src/virft/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed /work/docker/GRPO/Visual-RFT-main/src/virft/local_scripts/zero3.json \
    --max_prompt_length 5000 \
    --max_completion_length 7000 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --bf16 \
    --report_to wandb \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B_GRPO_coco_base65cate_6k \
    --save_steps 1 \
    --learning_rate=2e-06 \
    --save_only_model true \
    --num_generations 3 \

