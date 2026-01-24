# RealSR-R1: Reinforcement Learning for Real-World Image Super-Resolution with Vision-Language Chain-of-Thought

<div>
    Junbo Qiao<sup>1,†</sup>&emsp;
    Miaomiao Cai<sup>2†</sup>&emsp;
    Wei Li<sup>3*</sup>&emsp;
    Xudong Huang<sup>3</sup>&emsp;
    Jie Hu<sup>3</sup>&emsp;
    Xinghao Chen<sup>3</sup>&emsp;
    Shaohui Lin<sup>1*</sup>&emsp;
    Hongkai Xiong<sup>1,4</sup>&emsp;
</div>


<div>
    <sup>1</sup>East China Normal University, <sup>2</sup>University of Science and Technology of China, <sup>3</sup>Huawei Noah’s Ark Lab, <sup>4</sup>Shanghai Jiaotong University,<br/>
</div>

  <a href="https://www.arxiv.org/abs/2506.16796">
    <img
      src="https://img.shields.io/badge/RealSR_R1-paper-red?logo=arxiv&logoColor=red"
      alt="RealSR-R1 Paper on arXiv"
    />
   </a>

---

> **Abstract:** 
Real-World Image Super-Resolution is one of the most challenging task in image restoration. However, existing methods struggle with an accurate understanding of degraded image content, leading to reconstructed results that are both low-fidelity and unnatural. We present RealSR-R1, which empowers the RealSR models with understanding and reasoning capabilities. Inspired by the success of Chain of Thought (CoT) in large language models (LLMs), we simulate the human process of handling degraded images and propose the VLCoT framework, which integrates vision and language reasoning. The framework aims to precisely restore image details by progressively generating more comprehensive text and higher-resolution images. To overcome the challenge of traditional supervised learning CoT failing to generalize to real-world scenarios, we introduce, for the first time, Group Relative Policy Optimization (GRPO) into the Real-World Image Super-Resolution task. We propose VLCoT-GRPO as a solution, which designs four reward functions: (1) Format reward, used to standardize the CoT process; (2) Degradation reward, to incentivize accurate degradation estimation; (3) Understanding reward, to ensure the accuracy of the generated content; and (4) Generation reward, where we propose using a visual expert model to evaluate the quality of generated images, encouraging the model to generate more realistic images. Extensive experiments demonstrate that our proposed RealSR-R1 can generate realistic details and accurately understand image content, particularly in semantically rich scenes or images with severe degradation.

![RealSR-R1](./figs/fig1.png)

### Cold-start
Please refer to [Lumina-mGPT](https://github.com/Alpha-VLLM/Lumina-mGPT) for data processing and training code.


## 🛠️ Setup
```
git clone https://github.com/Junboooo/RealSR-R1.git
conda create -n RealSR-R1 python=3.10
conda activate RealSR-R1
bash setup.sh
```

### RL Dataset
Please refer to the script in ```./src/realsr-r1/create_json.py``` for dataset preparation. 
 
### VLCOTGRPO
After ready the dataset, you can start training using the following example bash script. Our bash scripts are in ```./src/scripts/example.py```
```
export DEBUG_MODE="true"
export LOG_PATH="./debug_log_2b_GRPO_coco_base65cate_6k.txt"

# export LD_LIBRARY_PATH=/opt/conda/envs/grpo/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# export LD_LIBRARY_PATH=/usr/local/cuda-12.1/compat:$LD_LIBRARY_PATH
# export CUDA_HOME=/usr/local/cuda

export DATA_PATH=/work/docker/GRPO/src/realsr-r1/test500_v2
export CKPT_PATH=/work/docker/GRPO/share_models/RealSR-R1_lr5e6/checkpoint-30 
export SAVE_PATH=./share_models/RealSR-R1_lr2e6

export PYTHONPATH=$PYTHONPATH:/work/docker/GRPO/src/realsr-r1
 #8192
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,6
export CUDA_LAUNCH_BLOCKING=1
torchrun --nproc_per_node="6" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="11317" \
    src/realsr-r1/src/open_r1/grpo.py \
    --output_dir ${SAVE_PATH}  \
    --model_name_or_path ${CKPT_PATH} \
    --dataset_name ${DATA_PATH} \
    --deepspeed /work/docker/GRPO/src/realsr-r1/local_scripts/zero3.json \
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
    --save_steps 1 \
    --learning_rate=2e-06 \
    --save_only_model true \
    --num_generations 5 \
```

## Results

<details>
<summary>Quantitative Comparisons (click to expand)</summary>

<p align="center">
  <img src="./figs/result.png">
</p>
</details>

<details>
<summary>Visual Comparisons (click to expand)</summary>

<p align="center">
  <img src="./figs/vis.png">
</p>
</details>


## Citation

If RealSR-R1 helps your research or work, please consider citing the following works:

----------
```BibTex
@inproceedings{qiao2024RealSR-R1,
  title={RealSR-R1: Reinforcement Learning for Real-World Image Super-Resolution with Vision-Language Chain-of-Thought},
  author={Qiao, Junbo and Cai, Miaomiao and Li, Wei and Liu, Yutong and Huang, Xudong and He, Gaoqi and Xie, Jiao and Hu, Jie and Chen, Xinghao and Lin, Shaohui},
  booktitle={arXiv preprint arXiv:2506.16796},
  year={2025}
}
```

## Acknowledgement
We sincerely thank projects <a href="https://github.com/Liuziyu77/Visual-RFT">Visual-RFT</a>,  <a href="https://github.com/nonwhy/PURE/">PURE</a> and <a href="https://github.com/Alpha-VLLM/Lumina-mGPT">Lumina-mGPT</a> for providing their open-source resources.
