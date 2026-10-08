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


import math
from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, \
                         LlamaConfig, LlamaModel, LlamaForCausalLM

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.generation.utils import GenerateOutput

from ..llava_arch import LlavaMetaModel, LlavaMetaForCausalLM
from ..router import Router, TextConditionedRouter, TextOnlyRouter


class ClipDinoCrossAttentionFusion(nn.Module):
    def __init__(self, hidden_size: int):
        super().__init__()
        self.clip_attn_q = nn.Linear(hidden_size, hidden_size)
        self.dino_attn_k = nn.Linear(hidden_size, hidden_size)
        self.dino_attn_v = nn.Linear(hidden_size, hidden_size)
        # we need to normalize the features. The 1/sqrt(hidden_size) attention
        # scale assumes the projected Q/K components have controlled variance.
        # Since CLIP/DINO projector outputs can have very different scales (max for
        #  without normalization. This causes instability in the dot product, which the scaling
        # (1/sqrt(hidden_size)) can not help, resulting in max of scores
        # being inf, and attn becoming NaN. Since clip_fetaures and dino_features
        # are outputs of MLP, we try and normalize using LayerNorm. We can not use RMSNorm
        # due to required torch version < 2.4.
        self.clip_features_norm = nn.LayerNorm(hidden_size)
        self.dino_features_norm = nn.LayerNorm(hidden_size)
        self.scale = 1.0 / (hidden_size ** 0.5)

    def forward(self, clip_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        # Run all fusion projections in one module call to keep ZeRO-3 param
        # gather/release aligned with module boundaries.
        # q is now much better after normalizing: max is 7.656 and std is 1.42 instead of >=4.
        clip_features = self.clip_features_norm(clip_features)
        dino_features = self.dino_features_norm(dino_features)
        q = self.clip_attn_q(clip_features) # max can reach 148 while min can reach -133 if we do not normalize
        k = self.dino_attn_k(dino_features) # max can reach 11 while min can reach -11. So clip is 10x if we do not normalize
        v = self.dino_attn_v(dino_features)
        scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.softmax(scores, dim=-1).to(device=dino_features.device, dtype=dino_features.dtype)
        return attn @ v


class ClipDinoFiLMFusion(nn.Module):
    """Use DINO patch features to FiLM-modulate CLIP patch features."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.clip_features_norm = nn.LayerNorm(hidden_size)
        self.dino_features_norm = nn.LayerNorm(hidden_size)
        self.film = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size * 2),
        )
        nn.init.zeros_(self.film[-1].weight)
        nn.init.zeros_(self.film[-1].bias)

    @staticmethod
    def _split_cls_token(features: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]:
        num_tokens = features.shape[1]
        side = math.isqrt(num_tokens)
        if side * side == num_tokens:
            return None, features

        patch_tokens = num_tokens - 1
        side = math.isqrt(patch_tokens)
        if patch_tokens > 0 and side * side == patch_tokens:
            return features[:, :1], features[:, 1:]

        return None, features

    @staticmethod
    def _infer_square_hw(num_tokens: int, feature_name: str) -> Tuple[int, int]:
        side = math.isqrt(num_tokens)
        if side * side != num_tokens:
            raise ValueError(
                f"ClipDinoFiLMFusion expects {feature_name} patch tokens to form a square grid, "
                f"but received {num_tokens} tokens."
            )
        return side, side

    def _resize_dino_to_clip_grid(
        self,
        dino_patches: torch.Tensor,
        clip_hw: Tuple[int, int],
    ) -> torch.Tensor:
        batch_size, num_dino_patches, hidden_size = dino_patches.shape
        dino_h, dino_w = self._infer_square_hw(num_dino_patches, "DINO")
        dino_grid = dino_patches.reshape(batch_size, dino_h, dino_w, hidden_size).permute(0, 3, 1, 2)
        interp_dtype = dino_grid.dtype
        dino_grid = F.interpolate(
            dino_grid.float(),
            size=clip_hw,
            mode="bilinear",
            align_corners=False,
        ).to(dtype=interp_dtype)
        return dino_grid.permute(0, 2, 3, 1).reshape(batch_size, clip_hw[0] * clip_hw[1], hidden_size)

    def forward(self, clip_features: torch.Tensor, dino_features: torch.Tensor) -> torch.Tensor:
        if isinstance(clip_features, (list, tuple)):
            if not isinstance(dino_features, (list, tuple)) or len(clip_features) != len(dino_features):
                raise ValueError("CLIP and DINO feature lists must have the same length for FiLM fusion.")
            return [self.forward(cur_clip, cur_dino) for cur_clip, cur_dino in zip(clip_features, dino_features)]

        squeeze_batch = clip_features.dim() == 2
        if squeeze_batch:
            clip_features = clip_features.unsqueeze(0)
            dino_features = dino_features.unsqueeze(0)

        if clip_features.dim() != 3 or dino_features.dim() != 3:
            raise ValueError(
                "ClipDinoFiLMFusion expects CLIP and DINO features shaped [B, N, H] or [N, H]."
            )

        clip_cls, clip_patches = self._split_cls_token(clip_features)
        _, dino_patches = self._split_cls_token(dino_features)
        clip_hw = self._infer_square_hw(clip_patches.shape[1], "CLIP")
        dino_aligned = self._resize_dino_to_clip_grid(dino_patches, clip_hw)
        dino_aligned = dino_aligned.to(device=clip_patches.device, dtype=clip_patches.dtype)

        film_input = torch.cat(
            [
                self.clip_features_norm(clip_patches),
                self.dino_features_norm(dino_aligned),
            ],
            dim=-1,
        )
        gamma, beta = self.film(film_input).chunk(2, dim=-1)
        fused_patches = clip_patches * (1.0 + gamma) + beta

        if clip_cls is not None:
            fused_features = torch.cat([clip_cls, fused_patches], dim=1)
        else:
            fused_features = fused_patches

        if squeeze_batch:
            fused_features = fused_features.squeeze(0)
        return fused_features


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
            hybrid_reserve_mode=getattr(config, "hybrid_reserve_mode", "local_gate"),
            hybrid_disagreement_metric=getattr(
                config, "hybrid_disagreement_metric", "kl_local_global"
            ),
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

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    encoder_type: str = "clip" # or dinov2_txt or clip-dinov2_txt
    fusion_type: str = "attn"
    fussion_type: str = "attn"


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        super(LlamaForCausalLM, self).__init__(config)

        self.model = LlavaLlamaModel(config)
        enc_type = getattr(config, 'encoder_type', '')
        if ("clip" in enc_type or "siglip" in enc_type) and "dino" in enc_type:
            fusion_type = getattr(config, 'fusion_type', 'attn')
            if fusion_type == "attn":
                self.model.clip_dino_fusion = ClipDinoCrossAttentionFusion(config.hidden_size)
            elif fusion_type == "film":
                self.model.clip_dino_fusion = ClipDinoFiLMFusion(config.hidden_size)

            if hasattr(self.model, "clip_dino_fusion"):
                rank0_print(f"Initialized {fusion_type} module for clip+dino fusion")
                rank0_print(f"params={sum(p.numel() for p in self.model.clip_dino_fusion.parameters())}")
        self.pretraining_tp = config.pretraining_tp
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

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
        dino_images: Optional[torch.FloatTensor] = None, # if we add dino
        image_sizes: Optional[List[List[int]]] = None,
        clip_texts: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
        encoder_type="clip",
        fusion_type=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # do I need a image_sizes_clip and a images_sizes_dino

        image_sizes_clip = None
        image_sizes_dino = None
        if image_sizes_clip is None and images is not None:
            image_sizes_clip = [list(list(x.shape[-2:])) for x in images]
        if image_sizes_dino is None and dino_images is not None:
            image_sizes_dino = [list(list(x.shape[-2:])) for x in dino_images]

        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels
            ) = self.prepare_inputs_labels_for_multimodal_withdino(
                input_ids=input_ids,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                labels=labels,
                images=images,
                dino_images=dino_images,
                image_sizes_clip=image_sizes_clip,
                image_sizes_dino=image_sizes_dino,
                clip_texts=clip_texts,
                dino_texts=clip_texts, # clip_texts is just the raw text
                encoder_type=encoder_type
            )
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
        dino_images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        encoder_type="clip",
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
        # clip_texts = kwargs.pop("clip_texts", None)
        if "inputs_embeds" in kwargs:
            raise NotImplementedError("`inputs_embeds` is not supported")

        image_sizes_clip = None
        image_sizes_dino = None
        if image_sizes_clip is None and images is not None:
            image_sizes_clip = [list(list(x.shape[-2:])) for x in images]
        if image_sizes_dino is None and dino_images is not None:
            image_sizes_dino = [list(list(x.shape[-2:])) for x in dino_images]

        if images is not None or dino_images is not None:
            (
                inputs,
                position_ids,
                attention_mask,
                _,
                inputs_embeds,
                _
            ) = self.prepare_inputs_labels_for_multimodal_withdino(
                input_ids=inputs,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=None,
                labels=None,
                images=images,
                dino_images=dino_images,
                image_sizes_clip=image_sizes_clip,
                image_sizes_dino=image_sizes_dino,
                clip_texts=prompts,
                dino_texts=prompts,
                encoder_type=encoder_type

            )
        else:
            inputs_embeds = self.get_model().embed_tokens(inputs)
        output = super().generate(
            position_ids=position_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            **kwargs
        )
        return output

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                      inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        dino_images = kwargs.pop("dino_images", None)
        image_sizes = kwargs.pop("image_sizes", None)
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        if images is not None:
            inputs['images'] = images
        if dino_images is not None:
            inputs['dino_images'] = dino_images
        if image_sizes is not None:
            inputs['image_sizes'] = image_sizes
        return inputs

class DoubleRouterLlavaLlamaForCausalLM(LlavaLlamaForCausalLM): # for both dino and clip
    def __init__(self, config, router_clip = None, router_dino = None, load_balance_alpha=10**-2, type="llava", aggregate_type = "attn"): # aggregate type will be one of linear, pool, attention.
                                                                                                                # linear means to concat then pass through linear layer,
                                                                                                                #  pool means to do line dino txt, and attention is
                                                                                                                # Jeonghwan's idea
        super().__init__(config)
        self.router_clip = router_clip
        self.router_dino = router_dino
        self.load_balance_alpha = load_balance_alpha
        self.aggregate_type = aggregate_type
        if aggregate_type=="linear":
            self.linear = nn.Linear(config.vision_hidden_size * 2, config.vision_hidden_size) # is there a way to define this here? Think I would need
                                                                                              #to get the hidden dims of clip and dino...
        self.text_embeds_clip = None
        self.text_embeds_dino = None
        self.text_attention_mask_clip = None
        self.text_attention_mask_dino = None

    def get_model(self):
        return self.model

    def rank0_print(self, *args):
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            print(*args)
    # the prepare_multimodal_inputs takes care of the merging
    def encode_images_withdino(self, dino_images, text_embeds=None, text_attention_mask=None):
        """
        text_embeds.shape,  text_attention_mask.shape
        (torch.Size([32, 2048]), torch.Size([32, 27]))
        """
        self.rank0_print(" \n ================= I AM IN ROUTER ENCODE DINO IMAGES =================")
        vision_tower = self.get_model().get_dino_vision_tower()
        device = next(vision_tower.parameters()).device
        dtype = next(vision_tower.parameters()).dtype
        self.text_embeds = text_embeds
        self.text_attention_mask = text_attention_mask

        if type(dino_images) is list:
            image_features = []
            for image in dino_images:
                _, hidden_states = vision_tower(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True)
                image_features.append(hidden_states)
            hidden_states = torch.stack([hs for hs in image_features])
        else:
            _,hidden_states = vision_tower(dino_images.to(device=device, dtype=dtype), output_hidden_states=True) # all 24 layers have  torch.Size([32, 257, 1024])
            hidden_states = torch.stack(list(hidden_states)) #torch.Size([24, 32, 256, 1024])
        assert hidden_states.ndim == 4 # torch.Size([24, 32, 256, 1024])
        image_features = self.router_dino(hidden_states, text_embeds=text_embeds, text_attention_mask=text_attention_mask, type="dino")
        image_features = self.get_model().dino_mm_projector(image_features)
        return image_features

    def encode_images_withclip(self, images, text_embeds=None, text_attention_mask=None, type_encoder="clip"):
        self.rank0_print(f" \n ================== I AM IN ROUTER ENCODE IMAGES WITH CLIP  using encoder {type_encoder}! ==================")
        vision_tower = self.get_model().get_vision_tower(load_model=type_encoder).vision_tower
        device = vision_tower.device
        dtype = vision_tower.dtype
        self.text_embeds = text_embeds
        self.text_attention_mask = text_attention_mask
        assert text_embeds is not None, "We are using text conditioned routing, so please pass in the text embedings"
        if type(images) is list:
            image_features = []
            for image in images:
                image_features.append(vision_tower(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True).hidden_states)
            hidden_states = torch.stack([hs for hs in image_features])
        else:
            image_forward_outs = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            hidden_states = torch.stack(list(image_forward_outs.hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]

        assert hidden_states.ndim == 4
        image_features = self.router_clip(hidden_states[1:, :, :, :], text_embeds=text_embeds, text_attention_mask=text_attention_mask, type=type_encoder) # do not want to include the embedding layer
        image_features = self.get_model().mm_projector(image_features) # Just one project?
        return image_features

    def get_router_probs_clip(self, hidden_states, encoder_type):
        hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        router_probs = self.router_clip.router_probs(hidden_states[1:, :, :, :], text_embeds=self.text_embeds_clip, text_attention_mask=self.text_attention_mask_clip, type=encoder_type) # remove embedding
        return router_probs

    def get_router_probs_dino(self, hidden_states):
        hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        router_probs = self.router_dino.router_probs(hidden_states, text_embeds=self.text_embeds_dino, text_attention_mask=self.text_attention_mask_dino) # dino I do not think embedding layer
        return router_probs

    def get_aux_loss(self, images, dino_images): #<- I need to pass in dino images
        # calculate the hidden states
        load_balanace_loss_clip = None
        load_blanace_loss_dino = None
        cached_router_probs_clip = getattr(self.router_clip, "_latest_router_probs_for_aux_clip", None)
        cached_router_probs_dino = getattr(self.router_dino, "_latest_router_probs_for_aux_dino", None)
        if cached_router_probs_clip is not None:
            load_balanace_loss_clip = self.router_clip.load_balance_loss(cached_router_probs_clip)
        if cached_router_probs_dino:
            load_blanace_loss_dino =  self.router_dino.load_balance_loss(cached_router_probs_dino)

        if load_balanace_loss_clip is not None and load_blanace_loss_dino is not None:
            loss = load_balanace_loss_clip + load_blanace_loss_dino
            return loss #TODO: change if needed

        vision_tower_clip = self.get_model().get_vision_tower().vision_tower
        if hasattr(self.get_model(), "get_dino_vision_tower"): # this works fine for seperate instances but not for when I have both. TODO: why?
            vision_tower_dino = self.get_model().get_dino_vision_tower().vision_tower
        else:
            vision_tower_dino = self.get_model().get_vision_tower(load_model="dino")
        device = vision_tower_clip.device
        dtype = vision_tower_clip.dtype
        if type(images) is list:
            losses_clip = []
            losses_dino = []
            for image, dino_image in zip(images, dino_images):
                image_forward_outs_clip = vision_tower_clip(image.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
                _, hidden_states = vision_tower_dino(dino_image.to(device=device, dtype=dtype), output_hidden_states=True)
                router_probs_clip = self.get_router_probs_clip(image_forward_outs_clip.hidden_states)
                router_probs_dino = self.get_router_probs_dino(hidden_states)
                losses_clip.append(self.router_clip.load_balance_loss(router_probs_clip))
                losses_dino.append(self.router_dino.load_balance_loss(router_probs_dino))
            loss = sum(losses_clip) / len(losses_clip) + sum(losses_dino) / len(losses_dino)
        else:
            image_forward_outs_clip = vision_tower_clip(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            _, hidden_states = vision_tower_dino(dino_images.to(device=device, dtype=dtype), output_hidden_states=True)
            router_probs_clip = self.get_router_probs_clip(image_forward_outs_clip.hidden_states)
            router_probs_dino = self.get_router_probs_dino(hidden_states)
            loss_clip = self.router_clip.load_balance_loss(router_probs_clip)
            loss_dino = self.router_dino.load_balance_loss(router_probs_dino)
            loss = loss_clip + loss_dino # how do I make this better? Do I introduce scaling coefs?
        return loss

class RouterLlavaLlamaForCausalLM(LlavaLlamaForCausalLM):
    """
    Router for a single encoder: clip or dino or maybe c4.
    """
    def __init__(self, config, router = None, load_balance_alpha=10**-2, type="llava", encoder_type=None):
        super().__init__(config)
        if encoder_type is None:
            encoder_type = getattr(config, 'encoder_type', 'clip')
        if "clip" in encoder_type or "siglip" in encoder_type:
            self.router_clip = router
        else:
            self.router_dino = router
        self.load_balance_alpha = load_balance_alpha
        self.encoder_type = encoder_type
        self.text_embeds = None
        self.text_attention_mask = None
        self.local_rank = None

        # you can using vision tower here because the vision tower is initialized
        # AFTER RouterLlavaLlamaForCausalLM.

    def rank0_print(self, *args):
        import os
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        if local_rank == 0:
            print(*args)

    def get_model(self):
        return self.model

    def encode_images_withdino(self, dino_images, text_embeds=None, text_attention_mask=None):
        """
        (Pdb) text_embeds.shape, text_attention_mask.shape
        (torch.Size([32, 2048]), torch.Size([32, 27]))
        """
        self.rank0_print(" \n ================= I AM IN ROUTER ENCODE DINO IMAGES =================")
        vision_tower = self.get_model().get_dino_vision_tower()
        device = next(vision_tower.parameters()).device
        dtype = next(vision_tower.parameters()).dtype
        self.text_embeds = text_embeds
        self.text_attention_mask = text_attention_mask

        if type(dino_images) is list:
            image_features = []
            for image in dino_images:
                _, hidden_states = vision_tower(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True)
                image_features.append(hidden_states)
            hidden_states = torch.stack([hs for hs in image_features])
        else:
            _,hidden_states = vision_tower(dino_images.to(device=device, dtype=dtype), output_hidden_states=True) # all 24 layers have  torch.Size([32, 257, 1024])
            hidden_states = torch.stack(list(hidden_states))
        assert hidden_states.ndim == 4
        image_features = self.router_dino(hidden_states, text_embeds=text_embeds, text_attention_mask=text_attention_mask, type="dino") # these are currently on cuda: 0
        image_features = self.get_model().dino_mm_projector(image_features)
        return image_features

    def encode_images_withclip(self, images, text_embeds=None, text_attention_mask=None, type_encoder="clip"):
        self.rank0_print(f" \n ================== I AM IN ROUTER ENCODE IMAGES WITH CLIP  using encoder {type_encoder}! ==================")
        vision_tower = self.get_model().get_vision_tower(load_model=type_encoder).vision_tower
        device = vision_tower.device
        dtype = vision_tower.dtype
        self.text_embeds = text_embeds
        self.text_attention_mask = text_attention_mask
        assert text_embeds is not None, "We are using text conditioned routing, so please pass in the text embedings"
        if type(images) is list:
            image_features = []
            for image in images:
                image_features.append(vision_tower(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True).hidden_states)
            hidden_states = torch.stack([hs for hs in image_features])
        else:
            image_forward_outs = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            hidden_states = torch.stack(list(image_forward_outs.hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]

        assert hidden_states.ndim == 4
        image_features = self.router_clip(hidden_states[1:, :, :, :], text_embeds=text_embeds, text_attention_mask=text_attention_mask, type=type_encoder) # do not want to include the embedding layer
        if torch.is_grad_enabled():
            assert image_features.requires_grad == True
        image_features = self.get_model().mm_projector(image_features) # Just one project?
        return image_features

    def get_router_probs(self, hidden_states):
        if not isinstance(hidden_states, (torch.Tensor)):
            hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        if self.encoder_type == "clip" or "siglip" in self.encoder_type:
            router_probs = self.router_clip.router_probs(hidden_states[1:, :, :, :], text_embeds=self.text_embeds, text_attention_mask=self.text_attention_mask, type=self.encoder_type) # remove embedding
        else:
            router_probs = self.router_dino.router_probs(hidden_states, text_embeds=self.text_embeds, text_attention_mask=self.text_attention_mask) # [24, 32, 257, 1024]
        return router_probs

    def get_aux_loss(self, images):
        # calculate the hidden states
        self.rank0_print("\n calculating aux_loss")
        cached_router_probs_clip = None
        cached_router_probs_dino = None

        if "clip" in self.encoder_type or "siglip" in self.encoder_type:
            cached_router_probs_clip = getattr(self.router_clip, "_latest_router_probs_for_aux_clip", None)
        if "dino" in self.encoder_type:
            cached_router_probs_dino = getattr(self.router_dino, "_latest_router_probs_for_aux_dino", None)

        if cached_router_probs_clip is not None:
            loss = self.router_clip.load_balance_loss(cached_router_probs_clip)
            return loss
        if cached_router_probs_dino is not None:
            loss = self.router_dino.load_balance_loss(cached_router_probs_dino)
            return loss

        if "clip" in self.encoder_type or "siglip" in self.encoder_type:
            if "clip" in self.encoder_type:
                load_model = "clip"
            else:
                load_model = "siglip"
            vision_tower = self.get_model().get_vision_tower(load_model = load_model).vision_tower
            device = vision_tower.device
            dtype = vision_tower.dtype
        else:
            vision_tower = self.get_model().get_vision_tower(load_model = "dino") # we write the forward which uses the vision tower, so better to just use class instance here
            device = self.get_model().device
            dtype = self.get_model().dtype
        losses = []
        if type(images) is list:
            losses = []
            for image in images:
                if "dino" in self.encoder_type:
                    image_forward_outs, hidden_states = vision_tower(image.to(device=device, dtype=dtype), output_hidden_states=True)
                else:
                    image_forward_outs = vision_tower(image.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
                    hidden_states = image_forward_outs.hidden_states
                router_probs = self.get_router_probs(hidden_states)
                if self.encoder_type == "clip" or "siglip" in self.encoder_type:
                    losses.append(self.router_clip.load_balance_loss(router_probs))
                else:
                    losses.append(self.router_dino.load_balance_loss(router_probs))
            loss = sum(losses) / len(losses)
        else:
            if "dino" in self.encoder_type:
                image_forward_outs, hidden_states = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True)
            else:
                image_forward_outs = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
                hidden_states = image_forward_outs.hidden_states
            router_probs = self.get_router_probs(hidden_states)
            if self.encoder_type == "clip" or "siglip" in self.encoder_type:
                loss = self.router_clip.load_balance_loss(router_probs)
            else:
                loss = self.router_dino.load_balance_loss(router_probs)
        return loss

AutoConfig.register("llava_llama", LlavaConfig)
AutoModelForCausalLM.register(LlavaConfig, LlavaLlamaForCausalLM, RouterLlavaLlamaForCausalLM)
AutoModelForCausalLM.register(LlavaConfig, DoubleRouterLlavaLlamaForCausalLM)
