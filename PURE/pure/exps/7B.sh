#!/bin/bash

lr=2e-5
wd=0.1
dropout=0.05
z_loss_weight=1e-5

data_config=configs/data/sample.yaml

exp_name=7B_ft_pix-r1
mkdir -p output/"$exp_name"

export PYTHONPATH=/work/docker/GRPO/PURE-main:$PYTHONPATH
export CUDA_LAUNCH_BLOCKING=1

CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7  torchrun --standalone --nproc_per_node=8 --nnodes=1 finetune_solver.py \
--model_size 7B \
--batch_size 2 \
--accum_iter 1 \
--epochs 2 \
--warmup_epochs 0.01 \
--lr ${lr} \
--min_lr ${lr} \
--wd ${wd} \
--clip_grad 4 \
--data_config $data_config \
--cache_ann_on_disk \
--num_workers 8 \
--output_dir output/"$exp_name" \
--save_iteration_interval 1000 \
--checkpointing \
--max_seq_len 11776 \
--unmask_image_logits \
--dropout ${dropout} \
--z_loss_weight ${z_loss_weight} \
2>&1 | tee -a output/"$exp_name"/output.log

echo "exp name: $exp_name"
