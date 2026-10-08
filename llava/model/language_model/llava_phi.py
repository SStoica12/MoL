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
from .llava_llama import _build_router_from_config, _config_requires_router


class LlavaPhiModel(LlavaMetaModel, PhiModel):
    config_class = PhiConfig

    def __init__(self, config: PhiConfig):
        super(LlavaPhiModel, self).__init__(config)


class LlavaPhiForCausalLM(PhiForCausalLM, LlavaMetaForCausalLM):
    config_class = PhiConfig

    def __init__(self, config):
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
        image_sizes: Optional[List[List[int]]] = None,
        clip_texts: Optional[List[str]] = None,
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
            ) = self.prepare_inputs_labels_for_multimodal(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                images=images,
                image_sizes=image_sizes,
                clip_texts=clip_texts,
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
        image_sizes: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        clip_texts = kwargs.pop("clip_texts", None)
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
            ) = self.prepare_inputs_labels_for_multimodal(
                inputs,
                position_ids,
                attention_mask,
                None,
                None,
                images,
                image_sizes=image_sizes,
                clip_texts=clip_texts,
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
    def __init__(self, config, router=None, load_balance_alpha=10**-2):
        super().__init__(config)
        if router is None and _config_requires_router(config):
            router = _build_router_from_config(self, config)
        self.router = router
        self.load_balance_alpha = load_balance_alpha

    def get_model(self):
        return self.model

    def encode_images(self, images, text_embeds=None, text_attention_mask=None):
        if self.router is None:
            return super().encode_images(
                images,
                text_embeds=text_embeds,
                text_attention_mask=text_attention_mask,
            )

        vision_tower = self.get_model().get_vision_tower().vision_tower
        device = vision_tower.device
        dtype = vision_tower.dtype
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_outs = vision_tower(
                    image.to(device=device, dtype=dtype).unsqueeze(0),
                    output_hidden_states=True,
                )
        else:
            image_forward_outs = vision_tower(
                images.to(device=device, dtype=dtype),
                output_hidden_states=True,
            )

        hidden_states = torch.stack(list(image_forward_outs.hidden_states))
        assert hidden_states.ndim == 4
        image_features = self.router(
            hidden_states[1:, :, :, :],
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
        )
        if self.training and torch.is_grad_enabled() and not image_features.requires_grad:
            raise RuntimeError("Router image features lost gradients during training.")
        mm_projector = self.get_model().mm_projector
        projector_param = next(mm_projector.parameters(), None)
        if projector_param is not None and (
            image_features.device != projector_param.device
            or image_features.dtype != projector_param.dtype
        ):
            image_features = image_features.to(
                device=projector_param.device,
                dtype=projector_param.dtype,
            )
        image_features = mm_projector(image_features)
        return image_features

    def get_router_probs(self, hidden_states):
        hidden_states = torch.stack(list(hidden_states))
        router_probs = self.router.router_probs(hidden_states[1:, :, :, :])
        return router_probs

    def get_aux_loss(self, images):
        cached_router_probs = getattr(self.router, "_latest_router_probs_for_aux", None)
        if cached_router_probs is not None:
            return self.router.load_balance_loss(cached_router_probs)

        vision_tower = self.get_model().get_vision_tower().vision_tower
        device = vision_tower.device
        dtype = vision_tower.dtype
        losses = []
        if type(images) is list:
            for image in images:
                image_forward_outs = vision_tower(
                    image.to(device=device, dtype=dtype),
                    output_hidden_states=True,
                )
                router_probs = self.get_router_probs(image_forward_outs.hidden_states)
                losses.append(self.router.load_balance_loss(router_probs))
            loss = sum(losses) / len(losses)
        else:
            image_forward_outs = vision_tower(
                images.to(device=device, dtype=dtype),
                output_hidden_states=True,
            )
            router_probs = self.get_router_probs(image_forward_outs.hidden_states)
            loss = self.router.load_balance_loss(router_probs)
        return loss
