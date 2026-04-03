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
#    See the License for the specific language governing permissions and
#    limitations under the License.


from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..router import Router, TextConditionedRouter, TextOnlyRouter


def _config_requires_router(config):
    if getattr(config, "has_router", False):
        return True
    architectures = getattr(config, "architectures", None) or []
    if any(str(arch).startswith("RouterLlava") for arch in architectures):
        return True
    if getattr(config, "use_text_conditioned_router", False):
        return True
    if getattr(config, "use_text_only_gating", False):
        return True
    return False


def _router_init_kwargs(config):
    return {
        "load_balancing_alpha": getattr(config, "load_balancing_alpha", 10**-2),
        "force_reserve_layer": getattr(config, "force_reserve_layer", False),
        "reserve_layer": getattr(config, "reserve_layer", -2),
        "learnable_gate": getattr(config, "learnable_gate", False),
        "confidence_gate": getattr(config, "confidence_gate", False),
        "confidence_gate_threshold": getattr(config, "confidence_gate_threshold", 0.6),
        "confidence_gate_slope": getattr(config, "confidence_gate_slope", 10.0),
        "vision_feature_select_strategy": getattr(config, "mm_vision_select_feature", "patch"),
    }


def _apply_router_runtime_settings(router, config, top_k):
    router.num_layers_to_choose = top_k
    router.save_router_stats = getattr(config, "save_router_stats", False)
    router.save_steps_router_stats = getattr(config, "save_steps_router_stats", 200)
    router.cap_softmax = getattr(config, "cap_softmax", False)
    router.cap_softmax_value = getattr(config, "cap_softmax_value", 0.06)
    return router


def _build_router_from_config(model, config):
    vision_tower = model.get_model().get_vision_tower()
    if vision_tower is None:
        raise ValueError("Router checkpoint requires a vision tower to reconstruct the router.")

    hidden_size = vision_tower.hidden_size
    num_image_tokens = vision_tower.num_patches + 1
    router_weights = torch.zeros(
        (num_image_tokens, hidden_size),
        dtype=model.lm_head.weight.dtype,
        device=model.lm_head.weight.device,
    )
    top_k = getattr(config, "num_layers_to_choose", 1)
    init_kwargs = _router_init_kwargs(config)

    if getattr(config, "use_text_only_gating", False):
        router = TextOnlyRouter(
            config,
            router_weights,
            top_k=top_k,
            text_pool=getattr(config, "text_pool", "mean"),
            text_scale=getattr(config, "text_scale", 1.0),
            **init_kwargs,
        )
    elif getattr(config, "use_text_conditioned_router", False):
        router = TextConditionedRouter(
            config,
            router_weights,
            top_k=top_k,
            text_pool=getattr(config, "text_pool", "mean"),
            text_scale=getattr(config, "text_scale", 1.0),
            text_conditioning=getattr(config, "text_conditioning", "film"),
            text_attention_type=getattr(config, "text_attention_type", "concat"),
            text_attention_dim=getattr(config, "text_attention_dim", None),
            attn_level=getattr(config, "attn_level", "layer"),
            **init_kwargs,
        )
    else:
        router = Router(
            config,
            router_weights,
            top_k=top_k,
            **init_kwargs,
        )

    return _apply_router_runtime_settings(router, config, top_k)


class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)
        self.model = LlavaLlamaModel(config)
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.model

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        images: Optional[torch.FloatTensor] = None,
        image_sizes: Optional[List[List[int]]] = None,
        clip_texts: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print("==================== I entered this forward ====================")
        # import sys; sys.exit(0)
        # import pdb; pdb.set_trace()
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
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
        # if torch.is_tensor(inputs_embeds):
        #     print("inputs_embeds: ", inputs_embeds.shape)
        # else:
        #     print("inputs_embeds: ", [x.shape for x in inputs_embeds.shape])
        # import sys; sys.exit()
        return super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict
        )

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
                _
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

        return super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

class RouterLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    def __init__(self, config, router = None, load_balance_alpha=10**-2, type="llava"):
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
                images, text_embeds=text_embeds, text_attention_mask=text_attention_mask
            )

        vision_tower = self.get_model().get_vision_tower().vision_tower
        device = vision_tower.device
        dtype = vision_tower.dtype
        if type(images) is list:
            image_features = []
            for image in images:
                image_forward_outs = vision_tower(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True)
        else:
            image_forward_outs = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])

        hidden_states = torch.stack(list(image_forward_outs.hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        assert hidden_states.ndim == 4
        image_features = self.router(
            hidden_states[1:, :, :, :],
            text_embeds=text_embeds,
            text_attention_mask=text_attention_mask,
        ) # do not want to include the embedding layer
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
        image_features = mm_projector(image_features) # image_features.requires_grad = True
        # assert image_features.requires_grad == True
        return image_features
    
    def get_router_probs(self, hidden_states):
        hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        router_probs = self.router.router_probs(hidden_states[1:, :, :, :]) # remove embedding
        return router_probs
    
    def get_aux_loss(self, images):
        # Prefer router probabilities from the latest multimodal forward pass.
        cached_router_probs = getattr(self.router, "_latest_router_probs_for_aux", None)
        if cached_router_probs is not None:
            return self.router.load_balance_loss(cached_router_probs)

        # Fallback path (legacy): recompute router probabilities from images.
        vision_tower = self.get_model().get_vision_tower().vision_tower
        device = vision_tower.device
        dtype = vision_tower.dtype
        losses = []
        if type(images) is list:
            losses = []
            for image in images:
                image_forward_outs = vision_tower(image.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
                router_probs = self.get_router_probs(image_forward_outs.hidden_states)
                losses.append(self.router.load_balance_loss(router_probs))
            loss = sum(losses) / len(losses)
        else:
            image_forward_outs = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            router_probs = self.get_router_probs(image_forward_outs.hidden_states)
            loss = self.router.load_balance_loss(router_probs)
        return loss

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, RouterLlavaLlamaForCausalLM)
