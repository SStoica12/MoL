#    Copyright 2023 Haotian Liu
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


import inspect
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import PhiConfig, PhiForCausalLM, PhiModel

from transformers.generation.utils import GenerateOutput
from transformers.modeling_outputs import CausalLMOutputWithPast

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from .llava_llama import (
    RouterLlavaLlamaForCausalLM,
    _build_router_from_config,
    _config_requires_router,
)


class LlavaPhiModel(LlavaMetaModel, PhiModel):
    config_class = PhiConfig

    def __init__(self, config: PhiConfig):
        super(LlavaPhiModel, self).__init__(config)


class LlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = PhiConfig

    def __init__(self, config):
        config.encoder_type = getattr(config, "encoder_type", "siglip")
        if config.encoder_type != "siglip":
            raise ValueError("The Phi model supports the single SigLIP encoder setup.")
        super(PhiForCausalLM, self).__init__(config)
        self.model = LlavaPhiModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=True)

        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        dino_images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        clip_texts: Optional[List[str]] = None,
        encoder_type: Optional[str] = None,
        fusion_type=None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        num_logits_to_keep: int = 0,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
            ) = self.prepare_inputs_labels_for_multimodal_withdino(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                images=images,
                dino_images=dino_images,
                image_sizes_clip=image_sizes,
                clip_texts=clip_texts,
                encoder_type=encoder_type or self.config.encoder_type,
            )

        super_forward = super().forward
        super_forward_params = inspect.signature(super_forward).parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in super_forward_params.values()
        )
        forward_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "position_ids": position_ids,
            "past_key_values": past_key_values,
            "inputs_embeds": inputs_embeds,
            "labels": labels,
            "use_cache": use_cache,
            "output_attentions": output_attentions,
            "output_hidden_states": output_hidden_states,
            "return_dict": return_dict,
        }
        if "cache_position" in super_forward_params:
            forward_kwargs["cache_position"] = cache_position
        if "num_logits_to_keep" in super_forward_params:
            forward_kwargs["num_logits_to_keep"] = num_logits_to_keep
        if accepts_kwargs:
            forward_kwargs.update(kwargs)
        else:
            forward_kwargs.update(
                {key: value for key, value in kwargs.items() if key in super_forward_params}
            )

        return super_forward(**forward_kwargs)

    @torch.no_grad()
    def generate(
        self,
        inputs: Optional[torch.Tensor] = None,
        images: Optional[torch.Tensor] = None,
        dino_images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        encoder_type: Optional[str] = None,
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        clip_texts = kwargs.pop("clip_texts", None)
        if prompts is not None:
            clip_texts = prompts
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        if images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _,
            ) = self.prepare_inputs_labels_for_multimodal_withdino(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                dino_images=dino_images,
                image_sizes_clip=image_sizes,
                clip_texts=clip_texts,
                encoder_type=encoder_type or self.config.encoder_type,
            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)

        if inputs is None and kwargs.get("bos_token_id", None) is None:
            bos_token_id = self.generation_config.bos_token_id
            if bos_token_id is None:
                bos_token_id = self.generation_config.eos_token_id
            if bos_token_id is None:
                bos_token_id = getattr(self.config, "pad_token_id", None)
            if bos_token_id is None:
                bos_token_id = 50256
            kwargs["bos_token_id"] = bos_token_id

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        **kwargs,
    ):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        super_prepare = super().prepare_inputs_for_generation
        super_prepare_params = inspect.signature(super_prepare).parameters
        accepts_kwargs = any(
            param.kind == inspect.Parameter.VAR_KEYWORD
            for param in super_prepare_params.values()
        )
        prepare_kwargs = {
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "inputs_embeds": inputs_embeds,
        }
        if "cache_position" in super_prepare_params:
            prepare_kwargs["cache_position"] = cache_position
        if accepts_kwargs:
            prepare_kwargs.update(kwargs)
        else:
            prepare_kwargs.update(
                {key: value for key, value in kwargs.items() if key in super_prepare_params}
            )
        inputs = super_prepare(input_ids, **prepare_kwargs)
        if images is not None:
            inputs["images"] = images
        if image_sizes is not None:
            inputs["image_sizes"] = image_sizes
        return inputs


class RouterLlavaPhiForCausalLM(LlavaPhiForCausalLM):
    def __init__(self, config, router=None, load_balance_alpha=10**-2, initialize_router=True):
        super().__init__(config)
        self.router_clip = router
        if router is None and initialize_router and _config_requires_router(config):
            self.initialize_router()
        self.load_balance_alpha = load_balance_alpha
        self.encoder_type = config.encoder_type
        self.text_embeds = None
        self.text_attention_mask = None

    def initialize_router(self):
        # SigLIP has no CLS token. Released Phi adapters have 27*27 positions.
        self.router_clip = _build_router_from_config(
            self, self.config, num_image_tokens=self.get_vision_tower().num_patches
        ).to(device=self.lm_head.weight.device, dtype=self.lm_head.weight.dtype)

    @classmethod
    def _load_pretrained_model(cls, model, *args, **kwargs):
        # Transformers 4.37 rewrites every "beta" in checkpoint keys to "bias".
        # Match that spelling during loading, then restore our canonical module
        # name. This preserves the trained hybrid gate in full-model exports.
        router = getattr(model, "router_clip", None)
        gate = getattr(router, "hybrid_beta_gate", None)
        if gate is None:
            return super()._load_pretrained_model(model, *args, **kwargs)
        del router.hybrid_beta_gate
        router.add_module("hybrid_bias_gate", gate)
        try:
            return super()._load_pretrained_model(model, *args, **kwargs)
        finally:
            gate = router.hybrid_bias_gate
            del router.hybrid_bias_gate
            router.add_module("hybrid_beta_gate", gate)

    # These methods depend only on the vision tower/router/projector interfaces.
    # Sharing them keeps Phi's routing and auxiliary loss identical to the
    # existing single-encoder implementation, without inheriting a Llama LM.
    rank0_print = RouterLlavaLlamaForCausalLM.rank0_print
    encode_images_withclip = RouterLlavaLlamaForCausalLM.encode_images_withclip
    get_router_probs = RouterLlavaLlamaForCausalLM.get_router_probs
    get_aux_loss = RouterLlavaLlamaForCausalLM.get_aux_loss
