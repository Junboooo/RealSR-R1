import argparse
import os
import sys
sys.path.append("./")

from inference_solver import FlexARInferenceSolver
from PIL import Image
from jacobi_iteration_pure import renew_pipeline_sampler
import torch
import time

import random
import numpy as np

def set_seed(seed: int):
    """
    Helper function for reproducible behavior to set the seed in random, numpy, torch.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)



# 创建函数来处理文件夹中的所有图像
def process_images(input_folder, image_output_folder, text_output_folder, inference_solver):
    # 获取文件夹中的所有图像文件
    for img_name in os.listdir(input_folder):
        # 只处理图像文件
        if img_name.endswith(('.png', '.jpg', '.jpeg')):
            img_path = os.path.join(input_folder, img_name)
            image = Image.open(img_path)

            # 提问的模板
            q1 = "Perceive the degradation level, understand the image content, and restore the high-quality image. <|image|>"
            qas = [[q1, None]]

            # 初始化推理模型
            inference_solver = renew_pipeline_sampler(
                inference_solver,
                jacobi_loop_interval_l=3,
                jacobi_loop_interval_r=(target_size // 8) ** 2 + target_size // 8 - 10,
                max_num_new_tokens=max_num_new_tokens,
                guidance_scale=guidance_scale,
                seed=None,
                multi_token_init_scheme=multi_token_init_scheme,
                do_cfg=True,
                text_top_k=text_top_k,
                prefix_token_sampler_scheme=prefix_token_sampler_scheme,
            )

            # 开始计时
            time_start = time.time()
            t1 = torch.cuda.Event(enable_timing=True)
            t2 = torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize()
            t1.record()

            # 生成结果
            generated = inference_solver.generate(
                images=[image],
                qas=qas,
                max_gen_len=11776,
                temperature=temperature,
                logits_processor=inference_solver.create_logits_processor(cfg=guidance_scale),
            )
            t2.record()
            torch.cuda.synchronize()

            time_elapsed = t1.elapsed_time(t2) / 1000
            time_end = time.time()
            print(f"Time elapsed for {img_name}: {time_elapsed} seconds. Total time: {time_end - time_start}")

            # 获取文本和生成图像
            text, new_image = generated[0], generated[1][0]

            # 保存图像到指定的文件夹
            new_image.save(os.path.join(image_output_folder, img_name), "PNG")

            # 保存文本到指定的文件夹
            text_file_path = os.path.join(text_output_folder, f"{os.path.splitext(img_name)[0]}.txt")
            with open(text_file_path, 'w') as f:
                f.write(text)

            print(f"Processed and saved: {img_name}, Text saved as {os.path.splitext(img_name)[0]}.txt")

if __name__ == '__main__':
    # 输入文件夹、输出文件夹
    input_folder = '/root/wx1233510/Lumina-mGPT-main/lumina_mgpt/test_dataset/benchmark_drealsr/test_LR' # 图像输入文件夹路径
    image_output_folder = '/root/wx1233510/PURE-main/PURE-main/result/benchmark_drealsr/PURE_image_jacobi'  # 输出图像保存文件夹
    text_output_folder = '/root/wx1233510/PURE-main/PURE-main/result/benchmark_drealsr/PURE_text_jasobi'  # 输出文本保存文件夹

    # 确保输出文件夹存在
    os.makedirs(image_output_folder, exist_ok=True)
    os.makedirs(text_output_folder, exist_ok=True)

    # 设定模型路径与目标大小
    model_path = "/root/wx1233510/PURE-main/PURE-main/pure/ckpts/Alpha-VLLM/PRUE"
    target_size = 512
    text_top_k = 1
    temperature = 0.9
    guidance_scale = 0.8

    max_num_new_tokens = 16
    multi_token_init_scheme = 'random'  # 'repeat_horizon'
    prefix_token_sampler_scheme = 'speculative_jacobi'  # 'jacobi', 'speculative_jacobi'

    # ******************** Image Generation ********************
    inference_solver = FlexARInferenceSolver(
        model_path=model_path,
        precision="bf16",
        target_size=target_size,
    )

    # 处理所有图像
    process_images(input_folder, image_output_folder, text_output_folder, inference_solver)
