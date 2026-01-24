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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AriaForConditionalGeneration,
    AriaProcessor,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Qwen2VLForConditionalGeneration,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available

from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.models import create_reference_model, prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url

import pyiqa
from torchvision import transforms
import json
import os
import torch
import copy

import sys
sys.path.append('/work/docker/GRPO/grpo-lumina/PURE/pure')
from inference_solver import FlexARInferenceSolver
from model.chameleon import ChameleonForConditionalGeneration
from PIL import Image
if is_peft_available():
    from peft import PeftConfig, get_peft_model
from sklearn.preprocessing import MinMaxScaler
if is_wandb_available():
    import wandb
import numpy as np
# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

# def iqa_reward(completions, TOPIQ,MUSIQ,MANIQA,CLIPIQA):
#     import pdb;pdb.set_trace()
#     topiq_value=[]
#     musiq_value=[]
#     maniqa_value=[]
#     clipiqa_value=[]
#     transform = transforms.Compose([transforms.ToTensor()])
#     device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
#     # 计算 MUSIQ 分数  
#     for i in range(len(completions)):
#         sr_img=completions[i][1][-1]
#         sr_img = transform(sr_img).unsqueeze(0).to(device)
#         # TOPIQ=pyiqa.create_metric('topiq_nr', device=device)
#         # CLIPIQA=pyiqa.create_metric('clipiqa', device=device)
#         # MUSIQ=pyiqa.create_metric('musiq', device=device)
#         # MANIQA=pyiqa.create_metric('maniqa-pipal', device=device)

#         with torch.no_grad():
#             topiq_value.append(TOPIQ(sr_img ).item())
#             musiq_value.append(MUSIQ(sr_img ).item())   
#             maniqa_value.append(MANIQA(sr_img ).item())
#             clipiqa_value.append(CLIPIQA(sr_img ).item())
#     # 将值转为 numpy 数组以便处理
#     topiq_value = np.array(topiq_value).reshape(-1, 1)
#     musiq_value = np.array(musiq_value).reshape(-1, 1)
#     maniqa_value = np.array(maniqa_value).reshape(-1, 1)
#     clipiqa_value = np.array(clipiqa_value).reshape(-1, 1)

#     # 使用 MinMaxScaler 对三个指标的值进行归一化
#     scaler = MinMaxScaler()

#     # 对每个值进行归一化
#     topiq_normalized = scaler.fit_transform(topiq_value)
#     musiq_normalized = scaler.fit_transform(musiq_value)
#     maniqa_normalized = scaler.fit_transform(maniqa_value)
#     clipiqa_normalized = scaler.fit_transform(clipiqa_value)

#     # 逐个位置相加，并计算每个位置的平均值
#     # import pdb;pdb.set_trace()
#     summed_values = topiq_normalized + musiq_normalized + maniqa_normalized +clipiqa_normalized
#     average_values = np.mean(summed_values, axis=1)

#     return  average_values

def iqa_reward(completions, TOPIQ,MUSIQ,MANIQA,CLIPIQA):
    # import pdb;pdb.set_trace()
    topiq_value=[]
    musiq_value=[]
    maniqa_value=[]
    clipiqa_value=[]
    transform = transforms.Compose([transforms.ToTensor()])
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # 计算 MUSIQ 分数  
    for i in range(len(completions)):
        sr_img=completions[i][1][-1]
        sr_img = transform(sr_img).unsqueeze(0).to(device)
        # TOPIQ=pyiqa.create_metric('topiq_nr', device=device)
        # CLIPIQA=pyiqa.create_metric('clipiqa', device=device)
        # MUSIQ=pyiqa.create_metric('musiq', device=device)
        # MANIQA=pyiqa.create_metric('maniqa-pipal', device=device)

        with torch.no_grad():
            topiq_value.append(TOPIQ(sr_img).item())
            musiq_value.append((MUSIQ(sr_img).item())/100)   
            maniqa_value.append(MANIQA(sr_img).item())
            clipiqa_value.append(CLIPIQA(sr_img).item())
    # 将值转为 numpy 数组以便处理
    topiq_value = np.array(topiq_value)
    musiq_value = np.array(musiq_value)
    maniqa_value = np.array(maniqa_value)
    clipiqa_value = np.array(clipiqa_value)

    # 使用 MinMaxScaler 对三个指标的值进行归一化
    # scaler = MinMaxScaler()

    # 对每个值进行归一化
    # topiq_normalized = scaler.fit_transform(topiq_value)
    # musiq_normalized = scaler.fit_transform(musiq_value)
    # maniqa_normalized = scaler.fit_transform(maniqa_value)
    # clipiqa_normalized = scaler.fit_transform(clipiqa_value)

    # 逐个位置相加，并计算每个位置的平均值
    # import pdb;pdb.set_trace()
    summed_values = topiq_value + musiq_value + maniqa_value +clipiqa_value
    average_values = summed_values/ 4


    return  average_values


class Qwen2VLGRPOTrainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):  
        self.TOPIQ=pyiqa.create_metric('topiq_nr', device="cuda")
        self.MUSIQ=pyiqa.create_metric('musiq', device="cuda")
        self.MANIQA=pyiqa.create_metric('maniqa-pipal', device="cuda")
        self.CLIPIQA=pyiqa.create_metric('clipiqa', device="cuda")
        self.stepp=0
        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        # Models
        # Trained model
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get("use_cache")
            )
            if "Qwen2-VL" in model_id:
                model = Qwen2VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "Qwen2.5-VL" in model_id:
                model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            elif "GRPO" in model_id:
                model = ChameleonForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
                
            elif "Aria" in model_id:
                model_init_kwargs.pop("use_cache")
                model = AriaForConditionalGeneration.from_pretrained(model, **model_init_kwargs)
            else:
                model = AutoModelForCausalLM.from_pretrained(model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        self.inference_solver = FlexARInferenceSolver(
            model_path=model, 
            precision="bf16",
            target_size=512,)
        
        ######################################
        # Reference model lzy modified
        # if peft_config is None:
        #     if is_deepspeed_zero3_enabled():
        #         if "Qwen2-VL" in model_id:
        #             self.ref_model = Qwen2VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #         elif "Qwen2.5-VL" in model_id:
        #             self.ref_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #         elif "PURE" in model_id:
        #             self.ref_model = ChameleonForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #         elif "Aria" in model_id:
        #             self.ref_model = AriaForConditionalGeneration.from_pretrained(model_id, **model_init_kwargs)
        #         else:
        #             self.ref_model = AutoModelForCausalLM.from_pretrained(model_id, **model_init_kwargs)
        #     else:
        #         # If PEFT configuration is not provided, create a reference model based on the initial model.
        #         self.ref_model = create_reference_model(model)
        # else:
        #     # If PEFT is used, the reference model is not needed since the adapter can be disabled
        #     # to revert to the initial model.
        #     self.ref_model = None
        ######################################
        # Processing class
        if processing_class is None:
            if "Qwen2-VL" in model_id or "Qwen2.5-VL" in model_id or "Aria" in model_id:
                processing_class = AutoProcessor.from_pretrained(model_id)
                pad_token_id = processing_class.tokenizer.pad_token_id
                processing_class.pad_token_id = pad_token_id
                processing_class.eos_token_id = processing_class.tokenizer.eos_token_id
                if "Qwen" in model_id or "Qwen2.5-VL" in model_id:
                    processing_class.image_processor.max_pixels = max_pixels
                    processing_class.image_processor.min_pixels = min_pixels
            else:
                processing_class = AutoTokenizer.from_pretrained(model.config._name_or_path, padding_side="left")
                pad_token_id = processing_class.pad_token_id
        
        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError("The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token
                # The reward model computes the reward for the latest non-padded token in the input sequence.
                # So it's important to set the pad token ID to the padding token ID of the processing class.
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes ###全是none？

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length  # = |o_i| in the GRPO paper
        self.num_generations = args.num_generations  # = G in the GRPO paper

        
        # self.generation_config = GenerationConfig(
        #     max_new_tokens=self.max_completion_length,
        #     do_sample=True,  
        #     temperature=1, # HACK
        #     num_return_sequences=self.num_generations,
        #     pad_token_id=pad_token_id,
        # )
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            max_length=self.inference_solver.model.config.max_position_embeddings,
            temperature=0.9,
            top_k=None,
            do_sample=True,
            num_return_sequences=self.num_generations,
            padding_idx=pad_token_id,
            # eos_token_id=[8710],
            # pad_token_id=pad_token_id,
            # num_return_sequences=2,
        )
        self.beta = args.beta

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        ######################################
        # if self.ref_model is not None:
        #     if self.is_deepspeed_enabled:
        #         self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
        #     else:
        #         self.ref_model = self.accelerator.prepare_model(self.ref_model, evaluation_mode=True)
        # #verl
        # for i, reward_func in enumerate(self.reward_funcs):
        #     if isinstance(reward_func, PreTrainedModel):
        #         self.reward_funcs[i] = self.accelerator.prepare_model(reward_func, evaluation_mode=True)
        ######################################
    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]


    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(self, model, input_ids, attention_mask):
        logits = model(input_ids, attention_mask=attention_mask).logits
        # logits = model(input_ids, attention_mask=attention_mask, pixel_values=pixel_values, image_grid_thw=image_grid_thw).logits  # (B, L, V)
        logits = logits[:, :-1, :]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[:, 1:]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(log_probs, dim=1, index=input_ids_row.unsqueeze(1)).squeeze(1)
            per_token_logps.append(token_log_prob)
        return torch.stack(per_token_logps)


    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
    
        ######################################################处理输入
        tags = [x["tag"] for x in inputs]
        images = [x["image"] for x in inputs]
        conversations = [x["conversations"] for x in inputs]
        prompts = [x["conversations"] for x in inputs]
        
        item =  [{"image": img, "conversations": conv} for img, conv in zip(images, conversations)]
        new_item = {'image': [x['image'] for x in item],'conversations': item[0]['conversations']}
        _prompt = self.inference_solver.item_processor.process_item(new_item)
        prompt = []
        for value in _prompt:
            if isinstance(value, int):
                prompt.append(value)
            else:
                prompt += value["input_ids"]
        prompt_len = len(prompt)
        prompt_ids = torch.tensor(prompt, dtype=torch.int64, device=self.model.device).unsqueeze(0)

        
        logits_processor = self.inference_solver.create_logits_processor(cfg=0.8, text_top_k=1)

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(prompt_ids, generation_config=self.generation_config, logits_processor=logits_processor, streamer=None, use_cache=True)
            # prompt_completion_ids = unwrapped_model.generate(**prompt_inputs, generation_config=self.generation_config) ###len=8 生成八次 # 生成提示补全ID
            prompt_length = prompt_ids.size(1) # 获取提示长度
            prompt_ids = prompt_completion_ids[:, :prompt_length] # 提取提示ID
            completion_ids = prompt_completion_ids[:, prompt_length:] # 提取补全ID

            # prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0) ##############################mask去掉了
        

        # # import pdb;pdb.set_trace()
        # # Mask everything after the first EOS token
        # # import pdb;pdb.set_trace()
        is_eos = completion_ids == 8710 #self.inference_solver.item_processor.process_item.tokenizer.eos_token_id # 在第一个EOS标记后屏蔽所有内容
        device = self.accelerator.device 
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device) # 初始化EOS索引
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)] # 更新EOS索引 
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1) # 生成序列索引
        # import pdb;pdb.set_trace()
        prompt_mask = torch.ones_like(prompt_ids)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int() # 生成补全掩码,补全到256，补全的为0，生成的为1

        # # # 将提示掩码与补全掩码连接以进行logit计算
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C) ##############################mask去掉了


        


        ######################################
        # #计算model的输入结果
        per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask)
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :] #8*256 计算输出结果的概率

        # with torch.inference_mode():
        #     if self.ref_model is not None:
        #         ref_per_token_logps = self._get_per_token_logps(self.ref_model, prompt_completion_ids, attention_mask=attention_mask) # 获取参考模型的每个token的log概率
        #     else:
        #         with self.accelerator.unwrap_model(model).disable_adapter():
        #             ref_per_token_logps = self._get_per_token_logps(model, prompt_completion_ids, attention_mask=attention_mask) # 获取模型的每个token的log概率  #8*900
        # ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :] #8*256 计算输出结果的概率
        ######################################
        # # 计算模型和参考模型之间的KL散度
        # per_token_kl = torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1


        completion_ids = completion_ids.tolist()
        # import pickle
        # # with open('/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/trainer/completion_ids.pkl', 'wb') as f: 
        # #     pickle.dump(completion_ids, f)
        # with open('/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/trainer/completion_ids.pkl', 'rb') as f:
        #     completion_ids = pickle.load(f)
        # # completion_ids = [completion_ids[0][:-1],completion_ids[1][:-1]]
        # print("############save##################")
        # import pdb;pdb.set_trace()
        completions = []


        for index, tokens in enumerate(completion_ids):  # Add index to differentiate folders
            generated, generated_images = self.inference_solver.decode_ids(tokens)
            completions.append((generated, generated_images))
            ###################################################
            # if self.state.global_step % 1 == 0 and self.accelerator.is_main_process:
            if self.accelerator.is_main_process:
                # Create a unique directory for each tokens based on the index
                step_dir = os.path.join(self.args.output_dir, f"step{self.state.global_step}", f"samples_{index}")
                os.makedirs(step_dir, exist_ok=True)  # Ensure the folder exists
                # Save the generated images
                for j, generated_image in enumerate(generated_images, 1):  # j starts from 1 for _1, _2, _3
                    image_filename = f"generated_image_{j}.png"  # Naming example: generated_image_1.png
                    image_path = os.path.join(step_dir, image_filename)
                    generated_image.save(image_path)  # Save the image

                # Save the generated text (for the current tokens)
                text_filename = "generated_text.txt"
                text_path = os.path.join(step_dir, text_filename)
                with open(text_path, 'w') as text_file:
                    text_file.write(generated)  # Save the text
            ##################################
        # with open('/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/trainer/completions_de.pkl', 'wb') as f: 
        #     pickle.dump(completions, f)
        # with open('/work/docker/GRPO/grpo-lumina/src/virft/src/open_r1/trainer/completions_de.pkl', 'rb') as f:
        #     completions = pickle.load(f)
        # import pdb;pdb.set_trace()
        # Compute the rewards
        
        ##############################################
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs)+1, device=device) # 初始化每个函数的奖励
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            # 模块而不是预训练模型以兼容编译模型
            if isinstance(reward_func, PreTrainedModel):
                print('nan')
            else:
                 # 重复所有输入列（但“prompt”和“completion”除外）以匹配生成的数量
                reward_kwargs = {key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]}
                for key in reward_kwargs: # 获取输入键 
                    for example in inputs:
                        # Repeat each value in the column for `num_generations` times
                        reward_kwargs[key].extend([example[key]] * self.num_generations) # 这里是把gt复制8份和预测的维度统一
                output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # 调用奖励函数 维度是8
                rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)# 转换奖励为张量
        
        # import pdb;pdb.set_trace()
        iqa_val=iqa_reward(completions, self.TOPIQ,self.MUSIQ,self.MANIQA,self.CLIPIQA)
        rewards_per_func[:, -1]=torch.tensor(iqa_val, dtype=torch.float32, device=device)
        # # 收集每个函数的奖励：这部分很关键，因为奖励是按组归一化的，补全可能分布在多个进程中
        # rewards = gather(rewards_per_func)  是将 rewards_per_func 张量从所有设备（如多个 GPU）收集到主设备（通常是第一个 GPU 或 CPU）
        # rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)  # 计算总奖励
        rewards = rewards_per_func.sum(dim=1)

        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)  #平均
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) #标准差

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # x - x.detach() allows for preserving gradients from x
        per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        # per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        per_token_loss = -(per_token_loss) #################
        loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()


        # Log the metrics
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # 记录指标，计算补全长度
        self._metrics["completion_length"].append(completion_length) # 记录补全长度

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(reward_per_func[i].item())
        # self._metrics[f"rewards/iqa"].append(iqa_val.item())

        self._metrics["reward"].append(self.accelerator.gather_for_metrics(rewards).mean().item())

        self._metrics["reward_std"].append(self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item())

        # mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        # self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        # 保存日志到文件###################################################
        if self.accelerator.is_main_process:
            log_file_path = os.path.join(self.args.output_dir, 'training_log.txt')
            with open(log_file_path, 'a') as f:
                f.write(f"Step: {self.state.global_step}, Loss: {loss.item()}, ")
                f.write(f"IQA: {iqa_val}")
                f.write(f"tag: {rewards_per_func[0]}")
                f.write(f"format: {rewards_per_func[1]}")
                f.write(f"reward: {rewards}")
                # for key, value in self._metrics.items():
                #     f.write(f"{key}: {value[-1]} ")  # 只保存当前训练步骤的最新值
                f.write("\n") 
        ######################################################################################################
        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val) for key, val in self._metrics.items()}  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available() and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))


