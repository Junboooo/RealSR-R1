# RealSR-R1: Reinforcement Learning for Real-World Image Super-Resolution with Vision-Language Chain-of-Thought

<div>
    Junbo Qiao<sup>1,†</sup>&emsp;
    Miaomiao Cai<sup>2†</sup>&emsp;
    Wei Li<sup>3*</sup>&emsp;
    Yutong Liu<sup>1,2</sup>&emsp;
    Xudong Huang<sup>3</sup>&emsp;
    Gaoqi He<sup>1</sup>&emsp;
    Jiao Xie<sup>1</sup>&emsp;
    Jie Hu<sup>3</sup>&emsp;
    Shaohui Lin<sup>1*</sup>&emsp;
</div>

<div>
    <sup>1</sup>East China Normal University, <sup>2</sup>University of Science and Technology of China, <sup>3</sup>Huawei Noah’s Ark Lab, <br/>
</div>

---

> **Abstract:** 
Real-World Image Super-Resolution is one of the most challenging task in image restoration. However, existing methods struggle with an accurate understanding of degraded image content, leading to reconstructed results that are both low-fidelity and unnatural. We present RealSR-R1, which empowers the RealSR models with understanding and reasoning capabilities. Inspired by the success of Chain of Thought (CoT) in large language models (LLMs), we simulate the human process of handling degraded images and propose the VLCoT framework, which integrates vision and language reasoning. The framework aims to precisely restore image details by progressively generating more comprehensive text and higher-resolution images. To overcome the challenge of traditional supervised learning CoT failing to generalize to real-world scenarios, we introduce, for the first time, Group Relative Policy Optimization (GRPO) into the Real-World Image Super-Resolution task. We propose VLCoT-GRPO as a solution, which designs four reward functions: (1) Format reward, used to standardize the CoT process; (2) Degradation reward, to incentivize accurate degradation estimation; (3) Understanding reward, to ensure the accuracy of the generated content; and (4) Generation reward, where we propose using a visual expert model to evaluate the quality of generated images, encouraging the model to generate more realistic images. Extensive experiments demonstrate that our proposed RealSR-R1 can generate realistic details and accurately understand image content, particularly in semantically rich scenes or images with severe degradation.

![RealSR-R1](./figs/fig1.png)

## ⚒️ TODO

* [ ] Release code and pretrained models
-

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
