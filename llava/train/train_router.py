# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    limitations under the License.
import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
from transformers import TrainerCallback
from llava.mm_utils import get_model_name_from_path
import torch
import wandb
import transformers
import tokenizers

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import RouterLLaVATrainer
from llava.model.router import Router, TextConditionedRouter, TextOnlyRouter
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token
from llava.model.language_model.llava_llama import RouterLlavaLlamaForCausalLM, DoubleRouterLlavaLlamaForCausalLM
from llava.model.language_model.llava_mpt import RouterLlavaMptForCausalLM, DoubleRouterLlavaMptForCausalLM
from PIL import Image


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    model_type: Optional[str] = field(default=None)
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_encoder_layers: bool = field(default=False)
    tune_residual_scaling_coef: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None) # will include scaling coefs
    pretrain_dino_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    vision_tower_delta: bool = field(default=False)
    num_layers_to_choose: Optional[int] = field(default=1)
    mm_use_text_enc: Optional[bool] = True
    use_text_conditioned_router: bool = False
    use_text_only_gating: bool = False
    text_pool: Optional[str] = "mean"
    text_scale: float = 1.0
    hybrid_beta_init: Optional[float] = None
    hybrid_beta_override: Optional[float] = None
    hybrid_reserve_mode: str = "local_gate"
    hybrid_disagreement_metric: str = "kl_local_global"
    text_conditioning: Optional[str] = "film"
    text_attention_type: Optional[str] = "concat"
    text_attention_dim: Optional[int] = 1024
    attn_level: Optional[str] = "layer"
    force_reserve_layer: bool = False
    reserve_layer : int = -2
    save_router_stats : bool = True
    save_steps_router_stats : int = 89
    cap_softmax: bool = False
    cap_softmax_value: float = 0.06
    learnable_gate: bool = False
    confidence_gate: bool = False
    confidence_gate_threshold: float = 0.6
    confidence_gate_slope: float = 10.0
    encoder_type: str = "clip"
    fusion_type: str = "attn" # options: attn, film, mean_pooling, concate_upsample


@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    freeze_residual_scaling_coef: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    dino_mm_projector_lr: Optional[float] = None
    router_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    load_balancing_alpha : float = 10**-2
    router_temp_start: float = 2.0
    router_temp_end: float = 1.0
    router_temp_schedule: str = field(default="linear")

class RouterCallback(TrainerCallback):
    def __init__(self, t_start: float, t_end: float, schedule: str = "linear", save_router_stats_clip=True, save_router_stats_dino=True, model=None, trainer=None):
        self.t_start = t_start
        self.t_end = t_end
        self.schedule = schedule
        self.last_temp = t_start
        self.save_router_stats_clip = save_router_stats_clip
        self.save_router_stats_dino = save_router_stats_dino

        self.model_ref = model
        self.trainer_ref = trainer

    def _compute_temp(self, step: int, max_steps: int) -> float:
        progress = min(step / max_steps, 1.0)
        if self.schedule == "linear":
            return self.t_start + (self.t_end - self.t_start) * progress
        if self.schedule == "cosine":
            return self.t_end + 0.5 * (self.t_start - self.t_end) * (1.0 + torch.cos(torch.tensor(progress * torch.pi)).item())
        if self.schedule == "exp":
            if self.t_start <= 0 or self.t_end <= 0:
                return self.t_end
            return self.t_start * ((self.t_end / self.t_start) ** progress)
        return self.t_end

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model", None)
        if model is None:
            model = self.model_ref
        if model is None:
            print(" CAN NOT FIND MODEL FOR LOADING")
            return
        m = model.module if hasattr(model, "module") else model
        if not hasattr(m, "router_clip") and not hasattr(m, "router_dino"):
            print("........ where did my router go?")
            return
        max_steps = max(state.max_steps, 1)
        t = self._compute_temp(state.global_step, max_steps)
        self.last_temp = t
        with torch.no_grad():
            if hasattr(m, "router_clip"):
                m.router_clip.router_temperature = t
            if hasattr(m, "router_dino"):
                m.router_dino.router_temperature = t

        # will remove later

    def on_log(self, args, state, control, logs=None, **kwargs):
        custom_metrics = {}
        if logs is None:
            print("logs is None.")
            return
        if not state.is_world_process_zero:
            return
        if wandb.run is None:
            return

        custom_metrics = {"router/callback_alive": 1}

        model = kwargs.get("model", None)
        if model is None:
            model = self.model_ref
        if model is None:
            return

        m = model.module if hasattr(model, "module") else model
        router_clip = getattr(m, "router_clip", getattr(getattr(m, "model", None), "router_clip", None))
        router_dino = getattr(m, "router_dino", getattr(getattr(m, "model", None), "router_dino", None))

        if (router_clip is None  or not self.save_router_stats_clip) and (router_dino is None or not self.save_router_stats_dino):
            return

        # pick the most recent step in your buffers (or drain them)
        if router_clip is not None and self.save_router_stats_clip:
            keys = sorted(router_clip.router_probs_per_step.keys())
            if not keys:
                return
            step_key = keys[-1]

            probs = router_clip.router_probs_per_step.pop(step_key)
            masks = router_clip.expert_mask_per_step.pop(step_key, None)
            hybrid_debug = getattr(router_clip, "hybrid_debug_per_step", {}).pop(step_key, None)

            avg_probs = probs.mean(dim=0).float().detach().cpu()
            for i, v in enumerate(avg_probs):
                custom_metrics[f"router_clip/prob_expert_{i}"] = float(v)

            if masks is not None:
                usage = masks.mean(dim=0).float().detach().cpu()
                for i, v in enumerate(usage):
                    custom_metrics[f"router_clip/usage_expert_{i}"] = float(v)

            if hybrid_debug is not None:
                custom_metrics["router_clip/disagreement_mean"] = float(
                    hybrid_debug["disagreement_mean"].item()
                )
                disagreement_per_layer = hybrid_debug["disagreement_per_layer_mean"].to(torch.float32)
                for i, value in enumerate(disagreement_per_layer):
                    custom_metrics[f"router_clip/disagreement_layer_{i}"] = float(value.item())

                custom_metrics["router_clip/beta_mean"] = float(
                    hybrid_debug["beta_mean"].item()
                )
                if "reserve_gamma_mean" in hybrid_debug:
                    custom_metrics["router_clip/reserve_gamma_mean"] = float(
                        hybrid_debug["reserve_gamma_mean"].item()
                    )

        custom_metrics["router_temperature"] = float(self.last_temp)
        wandb.log(custom_metrics)

        if router_dino is not None and self.save_router_stats_dino:
            keys = sorted(router_dino.router_probs_per_step.keys())
            if not keys:
                return
            step_key = keys[-1]

            probs = router_dino.router_probs_per_step.pop(step_key)
            masks = router_dino.expert_mask_per_step.pop(step_key, None)
            hybrid_debug = getattr(router_dino, "hybrid_debug_per_step", {}).pop(step_key, None)

            avg_probs = probs.mean(dim=0).float().detach().cpu()
            for i, v in enumerate(avg_probs):
                custom_metrics[f"router_dino/prob_expert_{i}"] = float(v)

            if masks is not None:
                usage = masks.mean(dim=0).float().detach().cpu()
                for i, v in enumerate(usage):
                    custom_metrics[f"router_dino/usage_expert_{i}"] = float(v)

            if hybrid_debug is not None:
                custom_metrics["router_dino/disagreement_mean"] = float(
                    hybrid_debug["disagreement_mean"].item()
                )
                disagreement_per_layer = hybrid_debug["disagreement_per_layer_mean"].to(torch.float32)
                for i, value in enumerate(disagreement_per_layer):
                    custom_metrics[f"router_dino/disagreement_layer_{i}"] = float(value.item())

                custom_metrics["router_dino/beta_mean"] = float(
                    hybrid_debug["beta_mean"].item()
                )
                if "reserve_gamma_mean" in hybrid_debug:
                    custom_metrics["router_dino/reserve_gamma_mean"] = float(
                        hybrid_debug["reserve_gamma_mean"].item()
                    )

        custom_metrics["router_temperature"] = float(self.last_temp)
        if self.trainer_ref is not None and hasattr(self.trainer_ref, "last_aux_loss"):
            m = self.trainer_ref.model.module if hasattr(self.trainer_ref.model, "module") else self.trainer_ref.model
            alpha = getattr(m, "load_balance_alpha", None)
            custom_metrics["router/aux_loss_raw"] = self.trainer_ref.last_aux_loss
            if alpha is not None:
                custom_metrics["router/aux_loss_weighted"] = self.trainer_ref.last_aux_loss * float(alpha)

        wandb.log(custom_metrics)

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return



def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(k == key or k.startswith(key + ".") or k == f"model.{key}" or k.startswith(f"model.{key}.") for key in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    multimodal_keywords = ['mm_projector', 'dino_mm_projector', 'vision_tower', 'vision_resampler', 'router', 'downsample_image_features']
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter and scaling coefs
        if hasattr(trainer.model, 'mm_projector') or hasattr(trainer.model.model, 'mm_projector'):
            keys_to_match = ['mm_projector', 'router_clip', 'clip_dino_fusion', 'downsample_image_features'] #'clip_attn_q', "dino_attn_k", "dino_attn_v"]
            if getattr(trainer.args, "tune_residual_scaling_coef", False):
                keys_to_match.extend(['scaling_coef'])
            if getattr(trainer.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            # keys_to_match.extend(['scaling_coef'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
                else:
                    torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))

        if hasattr(trainer.model, 'dino_mm_projector') or hasattr(trainer.model.model, 'dino_mm_projector'):
            keys_to_match = ['dino_mm_projector', 'router_dino']
            if getattr(trainer.args, "use_im_start_end", False):
                keys_to_match.extend(['embed_tokens', 'embed_in'])
            if getattr(trainer.args, "tune_residual_scaling_coef", False):
                keys_to_match.extend(['scaling_coef'])
            weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
            trainer.model.config.save_pretrained(output_dir)

            current_folder = output_dir.split('/')[-1]
            parent_folder = os.path.dirname(output_dir)
            if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
                if current_folder.startswith('checkpoint-'):
                    mm_projector_folder = os.path.join(parent_folder, "dino_mm_projector")
                    os.makedirs(mm_projector_folder, exist_ok=True)
                    torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
                else:
                    torch.save(weight_to_save, os.path.join(output_dir, f'dino_mm_projector.bin'))

        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir) # breaks here
        return
    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.mzdel_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources


def preprocess_llama_2(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.LLAMA_2

    # Mask targets
    sep = "[/INST] "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        clip_texts=conversations,
    )


def preprocess_v1(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    assert conv.sep_style == conversation_lib.SeparatorStyle.TWO

    # Mask targets
    sep = conv.sep + conv.roles[1] + ": "
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep2)
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 2
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 2

            if i != 0 and not tokenizer.legacy and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len -= 1
                instruction_len -= 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        clip_texts=conversations,
    )


def preprocess_mpt(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    conv = conversation_lib.default_conversation.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles[sentence["from"]]
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations

    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])] # system + user + gpt
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + gpt
        cur_len = 0
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
        clip_texts=conversations,
    )


def preprocess_plain(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        assert len(source) == 2
        assert DEFAULT_IMAGE_TOKEN in source[0]['value']
        source[0]['value'] = DEFAULT_IMAGE_TOKEN
        conversation = source[0]['value'] + source[1]['value'] + conversation_lib.default_conversation.sep
        conversations.append(conversation)
    # tokenize conversations
    input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_len = len(tokenizer_image_token(source[0]['value'], tokenizer))
        target[:tokenized_len] = IGNORE_INDEX

    return dict(input_ids=input_ids, labels=targets, clip_texts=conversations)


def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.PLAIN:
        return preprocess_plain(sources, tokenizer)
    if conversation_lib.default_conversation.sep_style == conversation_lib.SeparatorStyle.LLAMA_2:
        return preprocess_llama_2(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version.startswith("v1"):
        return preprocess_v1(sources, tokenizer, has_image=has_image)
    if conversation_lib.default_conversation.version == "mpt":
        return preprocess_mpt(sources, tokenizer, has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets, clip_texts=conversations)

def get_base_model(model_path, model_base, load_in_4bit, load_in_8bit, device_map):
    try:
        from llava.model.builder import load_pretrained_model
    except:
        raise ImportError("problem importing load_pretrained_model from llava.model.builder")
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path=model_path,
        model_base=model_base,
        model_name=model_name,
        load_8bit=load_in_8bit,
        load_4bit=load_in_4bit,
        device_map=device_map,
    )
    return tokenizer, model, image_processor, context_len

class ClipDinoDualProcessor(Dataset):
    """
    Takes data of type LazySupervisedDataset for both clip and
    dino and makes a single dataset containing both examples.
    Follows LazySupervisedDataset implementation
    """
    def __init__(self, train_dataset_clip, train_dataset_dino):
        super(ClipDinoDualProcessor, self).__init__()

        rank0_print("\n Creating combined Clip and Dino dataset for trainer")
        self.train_dataset_clip = train_dataset_clip
        self.train_dataset_dino = train_dataset_dino

    def __len__(self):
        return len(self.train_dataset_clip) # number of examples should just be
                                            # one since you are passing the data points as pairs
    @property
    def modality_lengths(self):
        return self.train_dataset_clip.modality_lengths


    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        item = self.train_dataset_clip[i]
        item['image_dino'] = self.train_dataset_dino[i]['image']
        return item

class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments,
                 for_model="clip"):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args
        self.for_model = for_model

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME

        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            image_folder = self.data_args.image_folder
            if self.for_model == "clip" or self.for_model == "siglip":
                processor = self.data_args.image_processor_clip # need to know when to process for dino and when to process for clip
            else:
                processor = self.data_args.image_processor_dino
            image = Image.open(os.path.join(image_folder, image_file)).convert('RGB')
            if self.data_args.image_aspect_ratio == 'pad':
                def expand2square(pil_img, background_color):
                    width, height = pil_img.size
                    if width == height:
                        return pil_img
                    elif width > height:
                        result = Image.new(pil_img.mode, (width, width), background_color)
                        result.paste(pil_img, (0, (width - height) // 2))
                        return result
                    else:
                        result = Image.new(pil_img.mode, (height, height), background_color)
                        result.paste(pil_img, ((height - width) // 2, 0))
                        return result
                image = expand2square(image, tuple(int(x*255) for x in processor.image_mean))
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            else:
                image = processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            has_image=('image' in self.list_data_dict[i]))
        if isinstance(i, int):
            clip_text = data_dict.get("clip_texts", None)
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])
            if clip_text is not None:
                data_dict["clip_texts"] = clip_text[0]

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            if self.for_model == "clip" or self.for_model == "siglip":
                processor = self.data_args.image_processor_clip # need to know when to process for dino and when to process for clip
                crop_size = processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
            else:
                processor = self.data_args.image_processor_dino
                crop_size = processor.crop_size
                data_dict['image'] = torch.zeros(3, crop_size, crop_size)
            # crop_size = processor.crop_size
            # data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'clip_texts' in instances[0]:
            batch['clip_texts'] = [instance['clip_texts'] for instance in instances]
        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            # with images in comput_loss in llava_trainer
            dino_images = [instance['image_dino'] for instance in instances if 'image_dino' in instance]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
                batch['dino_images'] = torch.stack(dino_images) if dino_images else None
            else:
                batch['images'] = images
                batch['dino_images'] = dino_images if dino_images else None

        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args, encoder_type) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = None
    if "clip" in encoder_type or "siglip" in encoder_type:
        train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args)
    train_dataset_dino = None
    if "dino" in encoder_type:
        train_dataset_dino = LazySupervisedDataset(tokenizer=tokenizer,
                                    data_path=data_args.data_path,
                                    data_args=data_args,
                                    for_model="dino")

    if train_dataset_dino is not None and train_dataset is not None:
        print("\n we are using combined dataset processor since we are using clip + dino instance. If this was not intended, please change the encoder_type parameter.")
        train_dataset = ClipDinoDualProcessor(train_dataset, train_dataset_dino)
    elif train_dataset_dino is not None:
        train_dataset = train_dataset_dino

    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)

    # we shoul not have a train_dataset and then a seperate train_dataset_dino because that
    # will mess up downstream since we use Trainer.
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train(attn_implementation=None):
    global local_rank
    rank0_print("====================== SUCCESSFULLY IN MOL TRAINER ROUTER ======================")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_skip_modules=["mm_projector", "router_dino", "router_clip", "dino_mm_projector", "clip_dino_fusion", "downsample_image_features"],
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))
    # tokenizer, model_base, image_processor_base, context_len = get_base_model(model_args.model_name_or_path, None, load_in_4bit=training_args.bits == 4, load_in_8bit=training_args.bits == 8, device_map=training_args.device)
    if model_args.vision_tower is not None:
        config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
        config.mm_hidden_size_dino = 1024
        config.fusion_type = model_args.fusion_type # this only matters if we have clip + dino instance

        config.encoder_type = model_args.encoder_type
        if 'mpt' in model_args.model_name_or_path:
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type)  and "dino" in model_args.encoder_type:
                model = DoubleRouterLlavaMptForCausalLM.from_pretrained(
                     model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
            else:
                model = RouterLlavaMptForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config,
                    cache_dir=training_args.cache_dir,
                    **bnb_model_from_pretrained_args
                )
                model.encoder_type = model_args.encoder_type
        else:
            # router is being passed in somewhere
            if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type)  and "dino" in model_args.encoder_type:
                model = DoubleRouterLlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config, # need to add this in or else it gets re-initialized
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
            else:
                model = RouterLlavaLlamaForCausalLM.from_pretrained(
                    model_args.model_name_or_path,
                    config=config, # need ot add this in or else it gets re-initialized
                    cache_dir=training_args.cache_dir,
                    attn_implementation=attn_implementation,
                    torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                    **bnb_model_from_pretrained_args
                )
                model.encoder_type = model_args.encoder_type
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            pretrained_model_name_or_pat=model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            config=config,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )

    model.config.use_cache = False
    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)
    model.to(device=training_args.device)
    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=model_args.model_type == "llama3-8b",
        )
    # tokenizer is of type transformers.models.llama.tokenization_llama.LlamaTokenizer
    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.model_type == "llama3-8b":
        tokenizer.eos_token_id = 128001
        if tokenizer.eos_token is None:
            tokenizer.eos_token = tokenizer.convert_ids_to_tokens(tokenizer.eos_token_id)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.eos_token_id = tokenizer.eos_token_id
        model.config.pad_token_id = tokenizer.pad_token_id

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        rank0_print("\n Vision tower initialized and frozen.")
        # initialize vision tower to frozen
        vision_tower = None
        dino_tower = None
        if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type):
            vision_tower = model.get_vision_tower()
            vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)
        if "dino" in model_args.encoder_type:
            dino_tower = model.get_dino_vision_tower() # create seperate dino_tower when we have clip + dino instance
            dino_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        model.config.force_reserve_layer = model_args.force_reserve_layer
        model.config.reserve_layer = model_args.reserve_layer
        model.config.save_router_stats = model_args.save_router_stats
        model.config.save_steps_router_stats = model_args.save_steps_router_stats
        model.config.cap_softmax = model_args.cap_softmax
        model.config.cap_softmax_value = model_args.cap_softmax_value
        model.config.learnable_gate = model_args.learnable_gate
        model.config.confidence_gate = model_args.confidence_gate
        model.config.confidence_gate_threshold = model_args.confidence_gate_threshold
        model.config.confidence_gate_slope = model_args.confidence_gate_slope
        model.config.use_text_conditioned_router = model_args.use_text_conditioned_router
        model.config.use_text_only_gating = model_args.use_text_only_gating
        model.config.text_conditioning = model_args.text_conditioning
        model.config.text_attention_type = model_args.text_attention_type
        model.config.text_attention_dim = model_args.text_attention_dim
        model.config.attn_level = model_args.attn_level
        model.config.text_pool = model_args.text_pool
        model.config.text_scale = model_args.text_scale
        model.config.hybrid_beta_init = model_args.hybrid_beta_init
        model.config.hybrid_beta_override = model_args.hybrid_beta_override
        model.config.hybrid_reserve_mode = model_args.hybrid_reserve_mode
        model.config.hybrid_disagreement_metric = model_args.hybrid_disagreement_metric
        model.config.num_layers_to_choose = model_args.num_layers_to_choose
        if model_args.hybrid_beta_init is not None:
            if not 0.0 < model_args.hybrid_beta_init < 1.0:
                raise ValueError("hybrid_beta_init must be strictly between 0 and 1.")
        if model_args.hybrid_beta_override is not None:
            if not 0.0 <= model_args.hybrid_beta_override <= 1.0:
                raise ValueError("hybrid_beta_override must be between 0 and 1.")
        if model_args.hybrid_beta_init is not None and model_args.hybrid_beta_override is not None:
            raise ValueError("hybrid_beta_init and hybrid_beta_override are mutually exclusive.")
        if model_args.hybrid_reserve_mode not in ("local_gate", "disagreement_gate"):
            raise ValueError(
                "hybrid_reserve_mode must be one of {'local_gate', 'disagreement_gate'}."
            )
        if model_args.hybrid_disagreement_metric not in ("kl_local_global",):
            raise ValueError(
                "hybrid_disagreement_metric must be 'kl_local_global'."
            )
        if (
            model_args.hybrid_reserve_mode == "disagreement_gate"
            and model_args.attn_level != "hybrid"
        ):
            raise ValueError(
                "hybrid_reserve_mode='disagreement_gate' requires attn_level='hybrid'."
            )
        if model_args.use_text_conditioned_router and model_args.use_text_only_gating:
            raise ValueError("use_text_conditioned_router and use_text_only_gating are mutually exclusive.")
        if (model_args.use_text_conditioned_router or model_args.use_text_only_gating) and getattr(model.config, "mm_text_hidden_size", None) is None:
            text_hidden_size = getattr(model.config, "hidden_size", None)
            if text_hidden_size is None:
                text_hidden_size = getattr(model.config, "d_model", None)
            if text_hidden_size is None and hasattr(model.get_model(), "embed_tokens"):
                text_hidden_size = model.get_model().embed_tokens.weight.shape[1]
            if text_hidden_size is None:
                raise ValueError("Unable to infer text hidden size for TextConditionedRouter.")
            model.config.mm_text_hidden_size = text_hidden_size

        if "dino" in model_args.encoder_type:
            model.config.dino_mm_text_hidden_size = 2048 # this is what is needed for dino


        rank0_print("\n Router configs set in model.config.")
        if dino_tower is not None: # initialize router_dino_weights
            image_size = dino_tower.config.image_size
            patch_size = dino_tower.config.patch_size
            hidden_size = dino_tower.config.hidden_size_dino
            num_image_tokens = (image_size // patch_size)**2 # we remove cls token
            router_dino_weights = torch.zeros((num_image_tokens, hidden_size)).to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device) # torch.Size([256, 1024])
        if vision_tower is not None:
            image_size = vision_tower.vision_tower.config.image_size
            patch_size = vision_tower.vision_tower.config.patch_size
            hidden_size = vision_tower.vision_tower.config.hidden_size
            num_image_tokens = (image_size // patch_size)**2
            if model_args.encoder_type == "clip":
                num_image_tokens += 1 # 257 even for dino txt, which has 4 registers
            router_clip_weights = torch.zeros((num_image_tokens, hidden_size)).to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device) #torch.Size([577, 1024])
        if not hasattr(model.config, 'mm_vision_select_feature'):
            model.config.mm_vision_select_feature = 'patch' # default or we get problems
        # router type
        if model_args.use_text_only_gating:
            router_cls = TextOnlyRouter
        elif model_args.use_text_conditioned_router:
            router_cls = TextConditionedRouter
        else:
            router_cls = Router
        if model_args.use_text_conditioned_router:
            if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type):
                # we use encoder type in text conditioning to infer the _infer_text_hidden_size dim,
                config_clip = copy.deepcopy(model.config)
                config_clip.encoder_type = "clip"
                model.router_clip = router_cls(
                    config_clip,
                    router_clip_weights,
                    config_clip.mm_vision_select_feature,
                    training_args.load_balancing_alpha,
                    text_pool=model_args.text_pool,
                    text_scale=model_args.text_scale,
                    text_conditioning=model_args.text_conditioning,
                    text_attention_type=model_args.text_attention_type,
                    text_attention_dim=model_args.text_attention_dim,
                    hybrid_reserve_mode=model_args.hybrid_reserve_mode,
                    hybrid_disagreement_metric=model_args.hybrid_disagreement_metric,
                    attn_level=model_args.attn_level,
                    force_reserve_layer=model_args.force_reserve_layer,
                    reserve_layer=model_args.reserve_layer,
                    learnable_gate=model_args.learnable_gate,
                    confidence_gate=model_args.confidence_gate,
                    confidence_gate_threshold=model_args.confidence_gate_threshold,
                    confidence_gate_slope=model_args.confidence_gate_slope,
                )
            if "dino" in model_args.encoder_type:
                config_dino = copy.deepcopy(model.config)
                config_dino.encoder_type = "dino"
                model.router_dino = router_cls(
                    config_dino,
                    router_dino_weights,
                    config_dino.mm_vision_select_feature,
                    training_args.load_balancing_alpha,
                    text_pool=model_args.text_pool,
                    text_scale=model_args.text_scale,
                    text_conditioning=model_args.text_conditioning,
                    text_attention_type=model_args.text_attention_type,
                    hybrid_reserve_mode=model_args.hybrid_reserve_mode,
                    hybrid_disagreement_metric=model_args.hybrid_disagreement_metric,
                    text_attention_dim=model_args.text_attention_dim,
                    attn_level=model_args.attn_level,
                    force_reserve_layer=model_args.force_reserve_layer,
                    reserve_layer=model_args.reserve_layer,
                    learnable_gate=model_args.learnable_gate,
                    confidence_gate=model_args.confidence_gate,
                    confidence_gate_threshold=model_args.confidence_gate_threshold,
                    confidence_gate_slope=model_args.confidence_gate_slope,
                )
        elif model_args.use_text_only_gating:
            if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type):
                model.router_clip = router_cls(
                    model.config,
                    router_clip_weights,
                    model.config.mm_vision_select_feature,
                    training_args.load_balancing_alpha,
                    text_pool=model_args.text_pool,
                    text_scale=model_args.text_scale,
                    force_reserve_layer=model_args.force_reserve_layer,
                    reserve_layer=model_args.reserve_layer,
                    learnable_gate=model_args.learnable_gate,
                    confidence_gate=model_args.confidence_gate,
                    confidence_gate_threshold=model_args.confidence_gate_threshold,
                    confidence_gate_slope=model_args.confidence_gate_slope,
                )
            if "dino" in model_args.encoder_type:
                model.router_dino = router_cls(
                    model.config,
                    router_dino_weights,
                    model.config.mm_vision_select_feature,
                    training_args.load_balancing_alpha,
                    text_pool=model_args.text_pool,
                    text_scale=model_args.text_scale,
                    force_reserve_layer=model_args.force_reserve_layer,
                    reserve_layer=model_args.reserve_layer,
                    learnable_gate=model_args.learnable_gate,
                    confidence_gate=model_args.confidence_gate,
                    confidence_gate_threshold=model_args.confidence_gate_threshold,
                    confidence_gate_slope=model_args.confidence_gate_slope,
                )
        else:
            if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type):
                model.router_clip = router_cls(
                    model.config,
                    router_clip_weights,
                    model.config.mm_vision_select_feature,
                    training_args.load_balancing_alpha,
                    learnable_gate=model_args.learnable_gate,
                    confidence_gate=model_args.confidence_gate,
                    confidence_gate_threshold=model_args.confidence_gate_threshold,
                    confidence_gate_slope=model_args.confidence_gate_slope,
                )
            if "dino" in model_args.encoder_type:
                model.router_dino = router_cls(
                    model.config,
                    router_dino_weights,
                    model.config.mm_vision_select_feature,
                    training_args.load_balancing_alpha,
                    learnable_gate=model_args.learnable_gate,
                    confidence_gate=model_args.confidence_gate,
                    confidence_gate_threshold=model_args.confidence_gate_threshold,
                    confidence_gate_slope=model_args.confidence_gate_slope,
                )
        # chooses which layers to apply the router
        if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type):
            model.router_clip.num_layers_to_choose = model_args.num_layers_to_choose
            model.router_clip.force_reserve_layer = model_args.force_reserve_layer
            model.router_clip.reserve_layer =  model_args.reserve_layer
            model.router_clip.save_steps_router_stats = model_args.save_steps_router_stats
            model.router_clip.save_router_stats = model_args.save_router_stats
            # make softmax less peaky
            model.router_clip.cap_softmax = model_args.cap_softmax
            model.router_clip.cap_softmax_value = model_args.cap_softmax_value
            model.router_clip.learnable_gate = model_args.learnable_gate
            model.router_clip.confidence_gate = model_args.confidence_gate
            model.router_clip.confidence_gate_threshold = model_args.confidence_gate_threshold
            model.router_clip.confidence_gate_slope = model_args.confidence_gate_slope
            if model.router_clip.learnable_gate and model.router_clip.reserve_gate is None:
                model.router_clip.reserve_gate = torch.nn.Linear(hidden_size, 1).to(device=training_args.device)
            model.router_clip.requires_grad_(True) # we need to learn the router
        if "dino" in model_args.encoder_type:
            model.router_dino.num_layers_to_choose = model_args.num_layers_to_choose
            model.router_dino.force_reserve_layer = model_args.force_reserve_layer
            model.router_dino.reserve_layer =  model_args.reserve_layer
            model.router_dino.save_steps_router_stats = model_args.save_steps_router_stats
            model.router_dino.save_router_stats = model_args.save_router_stats
            # make softmax less peaky
            model.router_dino.cap_softmax = model_args.cap_softmax
            model.router_dino.cap_softmax_value = model_args.cap_softmax_value
            model.router_dino.learnable_gate = model_args.learnable_gate
            model.router_dino.confidence_gate = model_args.confidence_gate
            model.router_dino.confidence_gate_threshold = model_args.confidence_gate_threshold
            model.router_dino.confidence_gate_slope = model_args.confidence_gate_slope
            if model.router_dino.learnable_gate and model.router_dino.reserve_gate is None:
                model.router_dino.reserve_gate = torch.nn.Linear(hidden_size, 1).to(device=training_args.device)
            model.router_dino.requires_grad_(True)

        if model_args.use_text_only_gating:
            num_layers_clip = None
            num_layers_dino = None

            num_layers_clip = getattr(vision_tower.vision_tower.config, "num_hidden_layers", None)
            num_layers_dino = getattr(dino_tower.config, "num_hidden_layers", None)

            if num_layers_clip is None and hasattr(vision_tower.vision_tower, "vision_model"):
                num_layers_clip = len(vision_tower.vision_tower.vision_model.encoder.layers)
            if num_layers_dino is None and hasattr(dino_tower, "vision_tower"):
                num_layers_dino = len(dino_tower.vision_tower.backbone.model.blocks)

            if num_layers_clip is None and num_layers_dino is None:
                raise ValueError("Unable to infer vision layer count for TextOnlyRouter.")

            if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type):
                model.router_clip._maybe_init_text_router(
                    num_layers_clip,
                    device=training_args.device,
                    dtype=router_clip_weights.dtype,
                )
            if "dino" in model_args.encoder_type:
                model.router_dino._maybe_init_text_router(
                    num_layers_dino,
                    device=training_args.device,
                    dtype=router_dino_weights.dtype,
                )

        if model_args.pretrain_mm_mlp_adapter is not None:
            weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                def remap(k): return '_weight_fn.' + k if k == 'weights' else k
                return {remap(k.split(keyword + '.')[1]): v for k, v in weights.items() if keyword in k}
            model.router_clip.load_state_dict(get_w(weights, 'router_clip'))
            rank0_print(" \n loaded router_clip weights...")

            fusion_module = getattr(model.get_model(), "clip_dino_fusion", None)
            uses_clip_dino_fusion = (
                model_args.fusion_type in ("attn", "film")
                and ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type)
                and "dino" in model_args.encoder_type
            )
            if uses_clip_dino_fusion:
                assert fusion_module is not None, (
                    f"fusion_type={model_args.fusion_type!r} requires clip_dino_fusion, but it was not initialized"
                )
            if fusion_module is not None:
                fusion_prefix = "model.clip_dino_fusion."
                fusion_w = {k.split(fusion_prefix, 1)[1]: v for k, v in weights.items() if fusion_prefix in k}
                if fusion_w:
                    missing, unexpected = fusion_module.load_state_dict(fusion_w, strict=False)
                    rank0_print(
                        f"Loaded {len(fusion_w)} pretrained clip_dino_fusion tensors "
                        f"with missing={missing}, unexpected={unexpected}"
                    )
                else:
                    rank0_print("No pretrained clip_dino_fusion weights found in pretrain_mm_mlp_adapter.")

        if model_args.pretrain_dino_mm_mlp_adapter is not None:
            weights = torch.load(model_args.pretrain_dino_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                def remap(k): return '_weight_fn.' + k if k == 'weights' else k
                return {remap(k.split(keyword + '.')[1]): v for k, v in weights.items() if keyword in k}
            model.router_dino.load_state_dict(get_w(weights, 'router_dino'))
            rank0_print(f"\n Successfully updated {get_w(weights, 'router_dino').keys()}")
        rank0_print("use_text_enc: ", model_args.mm_use_text_enc)

        data_args.image_processor_clip = None
        data_args.image_processor_dino = None
        if vision_tower is not None:
            data_args.image_processor_clip = vision_tower.image_processor
        if dino_tower is not None:
            data_args.image_processor_dino = dino_tower.image_processor

        if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type):
            assert data_args.image_processor_clip is not None
        if "dino" in model_args.encoder_type:
            assert data_args.image_processor_dino is not None
        data_args.is_multimodal = True


        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        model.config.tune_residual_scaling_coef = training_args.tune_residual_scaling_coef = model_args.tune_residual_scaling_coef

        # make the layers require gradients except the scaling coefs
        if model_args.tune_encoder_layers:
            if vision_tower is not None:
                layers = vision_tower.vision_tower.vision_model.encoder.layers
                for n,p in layers.named_parameters():
                    if 'scaling_coef' not in n:
                        p.requires_grad_(True)
            if dino_tower is not None:
                layers = dino_tower.vision_tower.backbone.model.blocks
                for n,p in layers.named_parameters():
                    if 'scaling_coef' not in n:
                        p.requires_grad_(True)

        fusion_module = getattr(model.get_model(), "clip_dino_fusion", None)
        uses_clip_dino_fusion = (
            model_args.fusion_type in ("attn", "film")
            and ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type)
            and "dino" in model_args.encoder_type
        )
        if uses_clip_dino_fusion:
            assert fusion_module is not None, (
                f"fusion_type={model_args.fusion_type!r} requires clip_dino_fusion, but it was not initialized"
            )
        if fusion_module is not None:
            for p in fusion_module.parameters():
                p.requires_grad = True

        if model_args.tune_mm_mlp_adapter:
            for n,p in model.named_parameters():
                if 'router' not in n:
                    p.requires_grad_(False)
            if hasattr(model, "router_dino"):
                model.router_dino.requires_grad_(True)
            if hasattr(model, "router_clip"):
                model.router_clip.requires_grad_(True)
            if hasattr(model.get_model(), "mm_projector"):
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = True

            if hasattr(model.get_model(), "dino_mm_projector"):
                for p in model.get_model().dino_mm_projector.parameters():
                        p.requires_grad = True
            if hasattr(model.get_model(), "clip_dino_fusion"):
                for p in model.get_model().clip_dino_fusion.parameters():
                    p.requires_grad = True

        if hasattr(model.get_model(), "downsample_image_features"):
            model.get_model().downsample_image_features.requires_grad_(True)

        if model_args.tune_residual_scaling_coef:
            if vision_tower is not None:
                for n, p in vision_tower.named_parameters():
                    if "scaling_coef" in n:
                        p.requires_grad = True
            if dino_tower is not None:
                for n, p in dino_tower.named_parameters():
                    if "scaling_coef" in n:
                        p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        model.config.freeze_residual_scaling_coef = training_args.freeze_residual_scaling_coef
        if training_args.freeze_mm_mlp_adapter:
            if hasattr(model.get_model(), "mm_projector"):
                for p in model.get_model().mm_projector.parameters():
                    p.requires_grad = False
            if hasattr(model.get_model(), "dino_mm_projector"):
                for p in model.get_model().dino_mm_projector.parameters():
                    p.requires_grad = False

        if training_args.freeze_mm_mlp_adapter and hasattr(model.get_model(), "downsample_image_features"):
            model.get_model().downsample_image_features.requires_grad_(False)

        if training_args.freeze_residual_scaling_coef:
            if vision_tower is not None:
                for n, p in vision_tower.named_parameters():
                    if "scaling_coef" in n:
                        p.requires_grad = False
            if dino_tower is not None:
                for n, p in dino_tower.named_parameters():
                    if "scaling_coef" in n:
                        p.requires_grad = False

        if training_args.bits in [4, 8]:
            if hasattr(model.get_model(), "mm_projector"):
                model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)
            if hasattr(model.get_model(), "dino_mm_projector"):
                model.get_model().dino_mm_projector.to(dtype=compute_dtype, device=training_args.device)
            if hasattr(model.get_model(), "clip_dino_fusion"):
                model.get_model().clip_dino_fusion.to(dtype=compute_dtype, device=training_args.device)
            if hasattr(model.get_model(), "downsample_image_features"):
                model.get_model().downsample_image_features.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        model.config.dino_mm_projector_lr = training_args.dino_mm_projector_lr
        model.config.router_lr = training_args.router_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        if (model_args.tune_residual_scaling_coef or model_args.tune_encoder_layers) and training_args.gradient_checkpointing: # this was needed because the vision model did not have inputs require True, resulting in no gradients being passed through the model
            if vision_tower is not None:
                if hasattr(vision_tower, "enable_input_require_grads"):
                    vision_tower.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    vision_tower.vision_tower.vision_model.embeddings.register_forward_hook(make_inputs_require_grad)

            if dino_tower is not None:
                if hasattr(dino_tower, "enable_input_require_grads"):
                    dino_tower.enable_input_require_grads()
                else:
                    def make_inputs_require_grad(module, input, output):
                        output.requires_grad_(True)
                    dino_tower.vision_tower.backbone.model.patch_embed.register_forward_hook(make_inputs_require_grad)
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args,
                                              encoder_type=model_args.encoder_type)
    trainer = RouterLLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    encoder_type=model_args.encoder_type,
                    fusion_type_when_preparing_inputs=model_args.fusion_type,
                    args=training_args,
                    **data_module)

    trainer.add_callback(
        RouterCallback(
            t_start=training_args.router_temp_start,
            t_end=training_args.router_temp_end,
            schedule=training_args.router_temp_schedule,
            save_router_stats_clip= model.router_clip.save_router_stats if (hasattr(model, 'router_clip') and model.router_clip is not None) else False,
            save_router_stats_dino= model.router_dino.save_router_stats if (hasattr(model, 'router_dino') and model.router_dino is not None) else False,
            model = model,
            trainer = trainer,
        )
    )
    trainable_params =  [n for n,p in model.named_parameters() if p.requires_grad]
    for n in trainable_params:
        assert 'scaling_coef' not in n
        assert 'vision_tower' not in n

    for n,p in model.named_parameters():
        if p.requires_grad:
            assert "vision_tower" not in n

    # ---- DIAGNOSTIC BLOCK: remove after confirming ----


    # ---- END DIAGNOSTIC BLOCK ----

    # ========================= UNCOMMENT ========================= #
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
        print("==================== RESUMING FROM CHECKPOINT ====================")
    else:
        trainer.train()
    # ========================= UNCOMMENT ========================= #
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

if __name__ == "__main__":
    train()
