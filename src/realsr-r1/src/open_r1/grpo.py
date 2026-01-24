# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from datasets import load_dataset, load_from_disk
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
# from open_r1.trainer import Qwen2VLGRPOTrainer
from open_r1.trainer import Qwen2VLGRPOTrainer, Qwen2VLGRPOVLLMTrainer
from trl import GRPOConfig, GRPOTrainer, ModelConfig, ScriptArguments, TrlParser, get_peft_config
from torchvision import transforms
import json
import os
import torch
import pyiqa
from PIL import Image

@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["accuracy", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )


def tag_reward(completions, tag, **kwargs):
    """Reward function that checks if tags appear in the <rough_understand> content and returns a score based on appearance ratio."""
    
    rough_understand_pattern = r"<rough_understand>(.*?)</rough_understand>"
    
    completion_contents = [completion[0] for completion in completions]
    rough_understand_contents = []
    
    for content in completion_contents:
        match = re.search(rough_understand_pattern, content, re.DOTALL)
        if match:
            rough_understand_contents.append(match.group(1).lower())
        else:
            rough_understand_contents.append("")  
    
    tag_appearance_ratios = []

    for i,content in enumerate(rough_understand_contents):
        tag_str = tag[i][0]
        tags = [item.strip() for item in tag_str.split(',') if item.strip()]
        total_tags = len(tags)
        matched_tags = 0 
        
        for t in tags:
            if t.lower() in content:
                matched_tags += 1
        
        ratio = matched_tags / total_tags
        tag_appearance_ratios.append(ratio)
    
    return [min(1.0, ratio) for ratio in tag_appearance_ratios]


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = (
        r"<degradation>.*?</degradation>\s*"
        r"<rough_understand>.*?</rough_understand>\s*"
        r"<rough_image>.*?</rough_image>\s*"
        r"<middle_understand>.*?</middle_understand>\s*"
        r"<middle_image>.*?</middle_image>\s*"
        r"<final_understand>.*?</final_understand>\s*"
        r"<final_image>.*?</final_image>"
    )
    
    completion_contents = [completion[0] for completion in completions]
    matches = [re.fullmatch(pattern, content, re.DOTALL) for content in completion_contents]
    return [1.0 if match else 0.0 for match in matches]


###  reward registry three parts
reward_funcs_registry = {
    "tag": tag_reward,
    # "iqa": iqa_reward,
    "format": format_reward,
}

SYSTEM_PROMPT = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
    "first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
    "<think> reasoning process here </think><answer> answer here </answer>"
)


def main(script_args, training_args, model_args):
    # Get reward functions
    script_args.reward_funcs = ['tag','format']
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]


    from datasets import DatasetDict
    dataset = load_dataset(script_args.dataset_name, name=script_args.dataset_config)

    def make_conversation(example):
        q1 = "Perceive the degradation, understand the image content, and restore the high-quality image step by step (simulating the image restoration process from coarse to fine). The low-quality image is as follows: <|image|>. The generation format should be as follows: <degradation> ... </degradation> <rough_understand> ... </rough_understand> <rough_image> ... </rough_image> <middle_understand> ... </middle_understand> <middle_image> ... </middle_image> <final_understand> ... </final_understand> <final_image> ... </final_image>"
        qas = [[q1, None]]
        
        conversations = []
        for q, a in qas:
            conversations.append(
                {
                    "from": "human",
                    "value": q,
                }
            )
            conversations.append(
                {
                    "from": "gpt",
                    "value": a,
                }
            )

        item = {
            "image": example["image"],
            "conversations": conversations, 
            "tag": example["tag"]
        }
        
        return item

    from datasets import DatasetDict
    dataset = DatasetDict.load_from_disk('/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/500_v2')
    dataset = dataset.map(make_conversation)
    
    trainer_cls = Qwen2VLGRPOTrainer if not training_args.use_vllm else Qwen2VLGRPOVLLMTrainer
    print("using: ", trainer_cls)

    # import pdb;pdb.set_trace()
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset= None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
    )
    trainer.train()

    # Save and push to hub
    print("save_model")
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # import pdb;pdb.set_trace()
    main(script_args, training_args, model_args)
