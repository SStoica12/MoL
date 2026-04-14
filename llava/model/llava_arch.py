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


from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from .multimodal_encoder.builder import build_vision_tower
from .multimodal_projector.builder import build_vision_projector

from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN

from llava.mm_utils import get_anyres_image_grid_shape

import torch.nn.functional as F

local_rank = None


def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

class LlavaMetaModel:

    def __init__(self, config):
        super(LlavaMetaModel, self).__init__(config)
        if hasattr(config, "mm_vision_tower"):
            # import pdb; pdb.set_trace()
            if "dino" in config.encoder_type:
                self.dino_tower = build_vision_tower(config, load_model = "dino", delay_load=True)
                self.dino_mm_projector = build_vision_projector(config, type="dino")
            if "siglip" in config.encoder_type:
                self.vision_tower = build_vision_tower(config, load_model = "siglip", delay_load=True)
                self.mm_projector = build_vision_projector(config, type="clip")
            if "siglip" not in config.encoder_type and "clip" in config.encoder_type:
                self.vision_tower = build_vision_tower(config, load_model = "clip", delay_load=True)
                self.mm_projector = build_vision_projector(config, type="clip")
            
            
        if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
            self.image_newline = nn.Parameter(
                torch.empty(config.hidden_size, dtype=self.dtype)
            )
        self.config = config
        self.encoder_type = config.encoder_type
        # if hasattr(config, "mm_vision_tower"):
        #     self.vision_tower = build_vision_tower(config, delay_load=True) # this will include scaling_coefs
        #     self.mm_projector = build_vision_projector(config)

        #     if 'unpad' in getattr(config, 'mm_patch_merge_type', ''):
        #         self.image_newline = nn.Parameter(
        #             torch.empty(config.hidden_size, dtype=self.dtype)
        #         )

    def get_dino_vision_tower(self):
        vision_tower = getattr(self, 'dino_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        assert vision_tower is not None
        return vision_tower

    def get_vision_tower(self, load_model = "clip"):
        # import pdb; pdb.set_trace()
        if load_model == "clip" or load_model == "siglip":
            vision_tower = getattr(self, 'vision_tower', None)
            if type(vision_tower) is list:
                vision_tower = vision_tower[0]

            return vision_tower

        vision_tower = getattr(self, 'dino_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]

        return vision_tower

        # get_dino_tower = getattr(self, 'dino_tower', None)
        # if get_dino_tower is None: # we only really have also dino_tower if we have clip_dino as encoder. Else, it can just be vision_tower
        #     vision_tower = getattr(self, 'vision_tower', None)
        #     if type(vision_tower) is list:
        #         vision_tower = vision_tower[0]

        #     return vision_tower
        # else:
        #     vision_tower = getattr(self, 'dino_tower', None)
        #     if type(vision_tower) is list:
        #         vision_tower = vision_tower[0]

        #     return vision_tower

    def initialize_vision_modules(self, model_args, fsdp=None):
        vision_tower = model_args.vision_tower
        mm_vision_select_layer = model_args.mm_vision_select_layer
        mm_vision_select_feature = model_args.mm_vision_select_feature
        pretrain_mm_mlp_adapter = None
        pretrain_dino_mm_mlp_adapter = None
        if ("clip" in model_args.encoder_type or "siglip" in model_args.encoder_type) :
            pretrain_mm_mlp_adapter = model_args.pretrain_mm_mlp_adapter
        if "dino" in self.encoder_type:
            pretrain_dino_mm_mlp_adapter = model_args.pretrain_dino_mm_mlp_adapter
        
        mm_patch_merge_type = model_args.mm_patch_merge_type

        self.config.mm_vision_tower = vision_tower
        self.config.mm_use_text_enc = model_args.mm_use_text_enc
        self.config.vision_tower_delta = getattr(model_args, "vision_tower_delta", False)
        
        if "siglip" in self.encoder_type:
            if self.get_vision_tower(load_model="siglip") is None:
                vision_tower = build_vision_tower(model_args, load_model="siglip")

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                else:
                    self.vision_tower = vision_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                else:
                    vision_tower = self.vision_tower
                vision_tower.load_model()

        if "clip" in self.encoder_type:
            if self.get_vision_tower(load_model="clip") is None:
                vision_tower = build_vision_tower(model_args)

                if fsdp is not None and len(fsdp) > 0:
                    self.vision_tower = [vision_tower]
                else:
                    self.vision_tower = vision_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    vision_tower = self.vision_tower[0]
                else:
                    vision_tower = self.vision_tower
                vision_tower.load_model()
        
        if "dino" in self.encoder_type:
            if  self.get_vision_tower(load_model="dino") is None:
                dino_tower = build_vision_tower(model_args, load_model="dino")

                if fsdp is not None and len(fsdp) > 0:
                    self.dino_tower = [dino_tower]
                else:
                    self.dino_tower = dino_tower
            else:
                if fsdp is not None and len(fsdp) > 0:
                    dino_tower = self.dino_tower[0]
                else:
                    dino_tower = self.dino_tower
                dino_tower.load_model()
        self.config.use_mm_proj = True
        self.config.mm_projector_type = getattr(model_args, 'mm_projector_type', 'linear') # can be ignored if using just Dino
        if "clip" in self.encoder_type or "siglip" in self.encoder_type:
            self.config.mm_hidden_size = vision_tower.hidden_size
        else:
            self.config.mm_hidden_size = self.config.hidden_size
        if ("clip" in self.encoder_type or "siglip" in self.encoder_type) and getattr(self.config, "mm_use_text_enc", False) and hasattr(vision_tower, "get_text_tower"):
            self.config.mm_text_hidden_size = vision_tower.get_text_tower().config.hidden_size
        if ("clip" in self.encoder_type or "siglip" in self.encoder_type):
            self.config.mm_vision_select_layer = mm_vision_select_layer
            self.config.mm_vision_select_feature = mm_vision_select_feature
        self.config.mm_patch_merge_type = mm_patch_merge_type

        if "clip" in self.encoder_type or "siglip" in self.encoder_type:
            if getattr(self, 'mm_projector', None) is None:
                self.mm_projector = build_vision_projector(self.config, type="clip") # may need to make siglip analog

                if 'unpad' in mm_patch_merge_type:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.image_newline = nn.Parameter(
                        torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                    )
            else:
                # In case it is frozen by LoRA
                for p in self.mm_projector.parameters():
                    p.requires_grad = True
                # infreeze also the scaling coefs if needed
                # for n, p in vision_tower.named_parameters():
                #     if "scaling_coef" in n:
                #         p.requires_grad = False

        if "dino" in self.encoder_type:
            if getattr(self, 'dino_mm_projector', None) is None:
                self.dino_mm_projector = build_vision_projector(self.config, type="dino")
                if 'unpad' in mm_patch_merge_type:
                    embed_std = 1 / torch.sqrt(torch.tensor(self.config.hidden_size, dtype=self.dtype))
                    self.image_newline = nn.Parameter(
                        torch.randn(self.config.hidden_size, dtype=self.dtype) * embed_std
                    )
            else:
                # in case it is frozen by llava
                for p in self.dino_mm_projector.parameters():
                    p.requires_grad = True

        if pretrain_mm_mlp_adapter is not None: # we assume this will only be for CLIP mm_proj
            weights = torch.load(pretrain_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.mm_projector.load_state_dict(get_w(weights, 'mm_projector'))
            rank0_print("============= Successfully updated pretrain_mm_mlp_adapter =============")
        # import pdb; pdb.set_trace()
        if pretrain_dino_mm_mlp_adapter is not None:
            rank0_print("\n Loading pretrained DinoV2 adpater...")
            dino_mm_projector_weights = torch.load(pretrain_dino_mm_mlp_adapter, map_location='cpu')
            def get_w(weights, keyword):
                return {k.split(keyword + '.')[1]: v for k, v in weights.items() if keyword in k}

            self.dino_mm_projector.load_state_dict(get_w(dino_mm_projector_weights, 'dino_mm_projector'))
            rank0_print("============= Successfully updated dino_mm_projector_weights =============")

def unpad_image(tensor, original_size):
    """
    Unpads a PyTorch tensor of a padded and resized image.

    Args:
    tensor (torch.Tensor): The image tensor, assumed to be in CxHxW format.
    original_size (tuple): The original size of PIL image (width, height).

    Returns:
    torch.Tensor: The unpadded image tensor.
    """
    original_width, original_height = original_size
    current_height, current_width = tensor.shape[1:]

    original_aspect_ratio = original_width / original_height
    current_aspect_ratio = current_width / current_height

    if original_aspect_ratio > current_aspect_ratio:
        scale_factor = current_width / original_width
        new_height = int(original_height * scale_factor)
        padding = (current_height - new_height) // 2
        unpadded_tensor = tensor[:, padding:current_height - padding, :]
    else:
        scale_factor = current_height / original_height
        new_width = int(original_width * scale_factor)
        padding = (current_width - new_width) // 2
        unpadded_tensor = tensor[:, :, padding:current_width - padding]

    return unpadded_tensor


class LlavaMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower() # what heppens if I have dino and clip?

    def get_dino_vision_tower(self):
        return self.get_model().get_vision_tower(load_model = "dino")
    
    def encode_images(self, images):
        image_features = self.get_model().get_vision_tower()(images) # image_features.requires_grad = False <-- this is why gradients are zero for scaling coefs
        assert image_features.requires_grad == True
        # import pdb; pdb.set_trace()
        image_features = self.get_model().mm_projector(image_features) # image_features.requires_grad = True
        return image_features

    
    def encode_images_withclip(self, images, text_embeds=None, text_attention_mask=None, type_encoder="clip"):
        rank0_print("\n Encoding images with NO ROUTER CLIP vision tower...")
        image_features_clip = self.get_model().get_vision_tower(load_model = type)(images)
        image_features_clip = self.get_model().mm_projector(image_features_clip)
        return image_features_clip


    def encode_images_withdino(self, images, text_embeds=None, text_attention_mask=None):
        # import pdb; pdb.set_trace()
        rank0_print("\n Encoding images with NO ROUTER DINO vision tower...")
        image_features_dino = self.get_model().get_vision_tower(load_model = "dino")(images)  # torch.Size([32, 256, 1024])
        # print("image_features_dino device", image_features_dino.device) # not reached.
        image_features_dino = self.get_model().dino_mm_projector(image_features_dino)
        return image_features_dino

    def prepare_inputs_labels_for_multimodal_withdino(
        self, 
        input_ids, 
        position_ids=None,
        attention_mask=None, 
        past_key_values=None, 
        labels=None, 
        images=None, 
        dino_images=None,
        image_sizes_clip=None,
        image_sizes_dino=None,
        clip_texts=None,
        dino_texts=None,
        encoder_type="clip"
        # encoder type can be one of clip, dino, clip_dino and interleave is only for clip_dino. If interleave is False, we use cross-attention 
    ):
        # import pdb; pdb.set_trace()
        dino_vision_tower = None
        vision_tower = None
        if "clip" in encoder_type or  "siglip" in encoder_type:
            vision_tower = self.get_vision_tower()
        if "dino" in encoder_type:
            dino_vision_tower = self.get_dino_vision_tower()

        if dino_images is not None and dino_vision_tower is None:
            raise ValueError("DINO images provided but no DINO vision tower available.")
        if dino_vision_tower is not None and dino_images is None:
            raise ValueError("DINO Tower provided but no dino images found.")
        if images is None and vision_tower is not None:
            raise ValueError("CLIP images not provided but CLIP vision tower available.")
        if vision_tower is None and images is not None:
            raise ValueError("CLIP images provided but CLIP vision tower not available.")

        if (vision_tower is None or images is None or input_ids.shape[1] == 1) and \
                (dino_vision_tower is None or dino_images is None or input_ids.shape[1] == 1):
            return input_ids, position_ids, attention_mask, past_key_values, None, labels
        
        text_embeds_clip = None
        text_embeds_dino = None
        text_attention_mask_clip = None
        text_attention_mask_dino = None
        # import pdb; pdb.set_trace()
        if getattr(self.config, "use_text_conditioned_router", False) or getattr(self.config, "use_text_only_gating", False):
            rank0_print("\n Using text conditioned routing...")
            use_clip_text = (
                clip_texts is not None
                and vision_tower is not None and hasattr(vision_tower, "text_tokenizer")
                and vision_tower is not None and hasattr(vision_tower, "encode_text")
            )
            if use_clip_text is False and ("clip" in encoder_type or "siglip" in encoder_type):
                raise ValueError(" \n we are not using clip_texts. Please pass in clip_texts.?")
            # print("\n use_clip_text = ", use_clip_text)
            use_dino_text = (
                dino_texts is not None
                and dino_vision_tower is not None and hasattr(dino_vision_tower, "text_tokenizer")
                and dino_vision_tower is not None and hasattr(dino_vision_tower, "encode_text")
            )
            if use_dino_text is False and "dino" in encoder_type:
                raise ValueError("\n we are not using dino_texts. Please pass in dino_texts")
            # import pdb; pdb.set_trace()
            # print("\n use_dino_text = ", use_dino_text)
            used_text = False # helps ensure we do not fall into else incorrectly if clip_text = True but dino_text is False
            if use_clip_text:
                used_text = True
                def _sanitize_clip_text(text):
                    if text is None:
                        return ""
                    text = text.replace(DEFAULT_IM_START_TOKEN, " ")
                    text = text.replace(DEFAULT_IM_END_TOKEN, " ")
                    text = text.replace(DEFAULT_IMAGE_PATCH_TOKEN, " ")
                    return " ".join(text.split())

                clip_texts = [_sanitize_clip_text(t) for t in clip_texts]
                if "siglip" in encoder_type:
                    clip_tokens = vision_tower.text_tokenizer(
                        clip_texts,
                        return_tensors="pt",
                        padding="max_length",
                        max_length=64,
                        truncation=True,
                        return_attention_mask=True,
                    )
                else:
                    clip_tokens = vision_tower.text_tokenizer(
                        clip_texts,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                    )
                clip_input_ids = clip_tokens["input_ids"].to(device=vision_tower.device)
                clip_attention_mask = clip_tokens.get("attention_mask", None)
                if clip_attention_mask is not None:
                    clip_attention_mask = clip_attention_mask.to(device=vision_tower.device)
                with torch.no_grad():
                    text_embeds_clip = vision_tower.encode_text(clip_input_ids, clip_attention_mask)
                text_attention_mask_clip = clip_attention_mask
            # import pdb;pdb.set_trace()
            if use_dino_text:
                used_text = True
                def _sanitize_dino_text(text):
                    if text is None:
                        return ""
                    text = text.replace(DEFAULT_IM_START_TOKEN, " ")
                    text = text.replace(DEFAULT_IM_END_TOKEN, " ")
                    text = text.replace(DEFAULT_IMAGE_PATCH_TOKEN, " ")
                    return " ".join(text.split())

                dino_texts = [_sanitize_dino_text(t) for t in dino_texts]
                dino_tokens = dino_vision_tower.text_tokenizer(
                    dino_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                dino_input_ids = dino_tokens["input_ids"].to(device=dino_vision_tower.device)
                dino_attention_mask = dino_tokens.get("attention_mask", None)
                if dino_attention_mask is not None:
                    dino_attention_mask = dino_attention_mask.to(device=dino_vision_tower.device)
                with torch.no_grad():
                    text_embeds_dino = dino_vision_tower.encode_text(dino_input_ids, dino_attention_mask)
                text_attention_mask_dino = dino_attention_mask
            # import pdb; pdb.set_trace()
            if not used_text:
                rank0_print("\n fallback to LLM embeddings instead of CLIPTextModel embeddings since CLIP text encoder is not available or clip_texts or dino_texts is None ...")
                # fallback to LLM embeddings instead of CLIPTextModel embeddings if CLIP text encoder is not available or clip_texts is None
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                text_attention_mask = attention_mask.bool()
                text_attention_mask = text_attention_mask & (input_ids != IMAGE_TOKEN_INDEX)
                if "clip" in encoder_type or "siglip" in encoder_type:
                    text_attention_mask_clip = text_attention_mask
                if "dino" in encoder_type:
                    text_attention_mask_dino = text_attention_mask
                pad_id = getattr(self.config, "pad_token_id", 0)
                safe_input_ids = input_ids.clone()
                safe_input_ids[safe_input_ids == IMAGE_TOKEN_INDEX] = pad_id
                text_embeds = self.get_model().embed_tokens(safe_input_ids)
                if "clip" in encoder_type or "siglip" in encoder_type:
                    text_embeds_clip = text_embeds
                if "dino" in encoder_type:
                    text_embeds_dino = text_embeds
                # text_embeds = self.get_model().embed_tokens(input_ids)
        # import sys; sys.exit(0)
        if images is None:
            condition = type(dino_images) is list or dino_images.ndim == 5
        else:
            condition = type(images) is list or images.ndim == 5
        if condition:
            
            if images is not None:
                if type(images) is list:
                    images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            
            if dino_images is not None:
                if type(dino_images) is list:
                    dino_images = [x.unsqueeze(0) if x.ndim == 3 else x for x in dino_images]
            
            if images is not None:
                concat_images = torch.cat([image for image in images], dim=0)
            if dino_images is not None:
                concat_dino_images = torch.cat([image for image in dino_images], dim=0)

            image_features_clip = None
            image_features_dino = None
            if "clip" in encoder_type or "siglip" in encoder_type:
                image_features_clip = self.encode_images_withclip(concat_images, text_embeds=text_embeds_clip, text_attention_mask=text_attention_mask_clip, type_encoder="siglip" if "siglip" in encoder_type else "clip")
            if "dino" in encoder_type:
                image_features_dino = self.encode_images_withdino(concat_dino_images, text_embeds=text_embeds_dino, text_attention_mask=text_attention_mask_dino)
            
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')

            if images is not None:
                split_sizes_clip = [image.shape[0] for image in images]
            if dino_images is not None:
                split_sizes_dino = [dino_image.shape[0] for dino_image in dino_images]
            
            if image_features_clip is not None:
                image_features_clip = torch.split(image_features_clip, split_sizes_clip, dim=0)
                if mm_patch_merge_type == 'flat':
                    image_features_clip = [x.flatten(0, 1) for x in image_features_clip]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_clip_image_features = []
                    for image_idx, image_feature in enumerate(image_features_clip):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = vision_tower.num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes_clip[image_idx], self.config.image_grid_pinpoints, vision_tower.config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes_clip[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_clip_image_features.append(image_feature)
                    image_features_clip = new_clip_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
            
            if image_features_dino is not None:
                image_features_dino = torch.split(image_features_dino, split_sizes_dino, dim=0)
                if mm_patch_merge_type == 'flat':
                    image_features_dino = [x.flatten(0, 1) for x in image_features_dino]
                elif mm_patch_merge_type.startswith('spatial'):
                    new_dino_image_features = []
                    for image_idx, image_feature in enumerate(image_features_dino):
                        if image_feature.shape[0] > 1:
                            base_image_feature = image_feature[0]
                            image_feature = image_feature[1:]
                            height = width = self.get_dino_vision_tower().num_patches_per_side
                            assert height * width == base_image_feature.shape[0]
                            if image_aspect_ratio == 'anyres':
                                num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes_dino[image_idx], self.config.image_grid_pinpoints, self.get_dino_vision_tower().config.image_size)
                                image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                            else:
                                raise NotImplementedError
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                                image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                                image_feature = unpad_image(image_feature, image_sizes_dino[image_idx])
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                                ), dim=-1)
                                image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                            else:
                                image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                                image_feature = image_feature.flatten(0, 3)
                            image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                        else:
                            image_feature = image_feature[0]
                            if 'unpad' in mm_patch_merge_type:
                                image_feature = torch.cat((
                                    image_feature,
                                    self.model.image_newline[None].to(image_feature.device)
                                ), dim=0)
                        new_dino_image_features.append(image_feature)
                    image_features_dino = new_dino_image_features
                else:
                    raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")

        else:
            # import pdb; pdb.set_trace()
            image_features_clip = None
            image_features_dino = None
            if "clip" in encoder_type or "siglip" in encoder_type:
                image_features_clip = self.encode_images_withclip(images, text_embeds=text_embeds_clip, text_attention_mask=text_attention_mask_clip, type_encoder="siglip" if "siglip" in encoder_type else "clip") # torch.Size([32, 576, 5120])
            if "dino" in encoder_type:
                image_features_dino = self.encode_images_withdino(dino_images, text_embeds=text_embeds_dino, text_attention_mask=text_attention_mask_dino) # torch.Size([32, 256, 1024])
        
        # import pdb; pdb.set_trace()

        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError
        
        
        _labels = labels # this is probably the tokenization of the text
        _position_ids = position_ids # None
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)
        #_input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)] # batch_size
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)] # batch_size


        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0

        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                # Use a zero-size slice for gradient connectivity
                if image_features_clip is not None: # think this was fine to be if and else even in clip + dino case
                    cur_image_features = image_features_clip[cur_image_idx]
                else:
                    cur_image_features = image_features_dino[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]]) # this is where the image is
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            # import pdb; pdb.set_trace()
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim)) # this is torch.Size([12, 5120])
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features_clip = image_features_clip[cur_image_idx] if image_features_clip is not None else None
                    cur_image_features_dino = image_features_dino[cur_image_idx] if image_features_dino is not None else None
                    cur_image_idx += 1
                    # at this point cur_image_features_clip is torch.Size([576, 5120]) while cur_image_features_dino is torch.Size([252, 5120])
                    # import pdb; pdb.set_trace()
                    if ("clip" in encoder_type or "siglip" in encoder_type) and "dino" in encoder_type and cur_image_features_clip is not None and cur_image_features_dino is not None:
                        # NOTE: we can not interleave because diff number of patches! Thus, we choose to concate and mean pool along 0th dimension.
                        # We can not follow dino txt implementation because it compresses dino features unto one token.
                        fusion_type = self.config.fusion_type # can be mean_pooling, concate_upsample or cross_attention
                        if fusion_type == "mean_pooling":
                            dim = cur_image_features_dino.shape[0]
                            if int(dim ** 0.5)**2 == dim: # we have perfect square
                                h_dim = w_dim = int(dim ** 0.5)
                            else:
                                h_dim = 14 # patch_size <- pass in
                                w_dim = dim // h_dim
                            assert h_dim * w_dim == dim, f"Cannot reshape DINO features of shape {cur_image_features_dino.shape} into a 2D grid for mean pooling"
                            cur_image_features_dino = cur_image_features_dino.reshape(1, cur_image_features_dino.shape[1], h_dim, w_dim) # torch.Size([1, 5120, 15, 15]) <-- avg 2d expects (bs, in_channels, H, W)
                            cur_image_features_dino = F.avg_pool2d(cur_image_features_dino, kernel_size=(2, 2))
                            cur_image_features_dino = cur_image_features_dino.reshape(-1, cur_image_features_dino.shape[1]) # torch.Size([63, 5120])
                            merged_features = torch.cat([cur_image_features_clip, cur_image_features_dino], dim=0) # [625, 5120]
                        
                        elif fusion_type == "concate_upsample" : # suggested by Jeonghwan
                            dim_clip = cur_image_features_clip.shape[0]
                            target_hight = target_width = int(dim_clip ** 0.5)
                            assert target_hight * target_width == dim_clip, f"Cannot reshape CLIP features of shape {cur_image_features_clip.shape} into a 2D grid for upsample concat fusion"
                            # want to torch.Size([252, 5120]) --> torch.Size([dim, 5120])
                            dim = cur_image_features_dino.shape[0]
                            if int(dim ** 0.5)**2 == dim: # we have perfect square
                                h_dim = w_dim = int(dim ** 0.5)
                            else:
                                h_dim = 14 # patch_size <- pass in
                                w_dim = dim // h_dim
                            assert h_dim * w_dim == dim, f"Cannot reshape DINO features of shape {cur_image_features_dino.shape} into a 2D grid for mean pooling"
                            # interpolate dino features to target hight and width
                            cur_image_features_dino = cur_image_features_dino.reshape(1, cur_image_features_dino.shape[1], h_dim, w_dim) # torch.Size([1, 5120, 15, 15])
                            cur_image_features_dino = F.interpolate(
                                cur_image_features_dino,
                                size=(target_hight, target_width),
                                mode="bilinear",
                                align_corners=False,
                            )
                            cur_image_features_dino = cur_image_features_dino.reshape(cur_image_features_clip.shape) # torch.Size([252, 5120])
                            merged_features = torch.cat([cur_image_features_clip, cur_image_features_dino], dim=-1)
                        else:
                            # Cross-attention fusion: Q=CLIP, K=V=DINOv2
                            Q = cur_image_features_clip
                            K = cur_image_features_dino
                            V = cur_image_features_dino
                            # [576, 5120] @ [5120, 252] = [576, 252] @ [252, 5120] = [576, 5120]
                            merged_features = torch.softmax(Q @ K.transpose(-2, -1) / (K.shape[-1] ** 0.5), dim=-1) @ V # [576, 5120]
                        cur_new_input_embeds.append(merged_features)
                        cur_new_labels.append(torch.full((merged_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif cur_image_features_clip is not None:
                        cur_new_input_embeds.append(cur_image_features_clip)
                        cur_new_labels.append(torch.full((cur_image_features_clip.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                    elif cur_image_features_dino is not None:
                        cur_new_input_embeds.append(cur_image_features_dino) # torch.Size([256, 1024])
                        cur_new_labels.append(torch.full((cur_image_features_dino.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
            # import pdb; pdb.set_trace()
            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]
            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)
            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)


        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad and build final tensors
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels
    

    def prepare_inputs_labels_for_multimodal(
        self,
        input_ids,
        position_ids=None,
        attention_mask=None,
        past_key_values=None,
        labels=None,
        images=None,
        image_sizes=None,
        clip_texts=None,
    ):
        vision_tower = self.get_vision_tower()
        if vision_tower is None or images is None or input_ids.shape[1] == 1:
            return input_ids, position_ids, attention_mask, past_key_values, None, labels

        text_embeds = None
        text_attention_mask = None
        if getattr(self.config, "use_text_conditioned_router", False) or getattr(self.config, "use_text_only_gating", False):
            # breakpoint() - TODO: Why isn't `use_clip_text` set to True?
            use_clip_text = (
                clip_texts is not None
                and hasattr(vision_tower, "text_tokenizer")
                and hasattr(vision_tower, "encode_text")
            )
            if use_clip_text:
                def _sanitize_clip_text(text):
                    if text is None:
                        return ""
                    text = text.replace(DEFAULT_IM_START_TOKEN, " ")
                    text = text.replace(DEFAULT_IM_END_TOKEN, " ")
                    text = text.replace(DEFAULT_IMAGE_PATCH_TOKEN, " ")
                    return " ".join(text.split())

                clip_texts = [_sanitize_clip_text(t) for t in clip_texts]
                clip_tokens = vision_tower.text_tokenizer(
                    clip_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                )
                clip_input_ids = clip_tokens["input_ids"].to(device=vision_tower.device)
                clip_attention_mask = clip_tokens.get("attention_mask", None)
                if clip_attention_mask is not None:
                    clip_attention_mask = clip_attention_mask.to(device=vision_tower.device)
                with torch.no_grad():
                    text_embeds = vision_tower.encode_text(clip_input_ids, clip_attention_mask)
                text_attention_mask = clip_attention_mask
            else:
                # fallback to LLM embeddings instead of CLIPTextModel embeddings if CLIP text encoder is not available or clip_texts is None
                if attention_mask is None:
                    attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
                text_attention_mask = attention_mask.bool()
                text_attention_mask = text_attention_mask & (input_ids != IMAGE_TOKEN_INDEX)
                pad_id = getattr(self.config, "pad_token_id", 0)
                safe_input_ids = input_ids.clone()
                safe_input_ids[safe_input_ids == IMAGE_TOKEN_INDEX] = pad_id
                text_embeds = self.get_model().embed_tokens(safe_input_ids)
                # text_embeds = self.get_model().embed_tokens(input_ids)

        if type(images) is list or images.ndim == 5:
            if type(images) is list:
                images = [x.unsqueeze(0) if x.ndim == 3 else x for x in images]
            concat_images = torch.cat([image for image in images], dim=0)
            image_features = self.encode_images(
                concat_images, text_embeds=text_embeds, text_attention_mask=text_attention_mask
            )
            split_sizes = [image.shape[0] for image in images]
            image_features = torch.split(image_features, split_sizes, dim=0)
            mm_patch_merge_type = getattr(self.config, 'mm_patch_merge_type', 'flat')
            image_aspect_ratio = getattr(self.config, 'image_aspect_ratio', 'square')
            if mm_patch_merge_type == 'flat':
                image_features = [x.flatten(0, 1) for x in image_features]
            elif mm_patch_merge_type.startswith('spatial'):
                new_image_features = []
                for image_idx, image_feature in enumerate(image_features):
                    if image_feature.shape[0] > 1:
                        base_image_feature = image_feature[0]
                        image_feature = image_feature[1:]
                        height = width = self.get_vision_tower().num_patches_per_side
                        assert height * width == base_image_feature.shape[0]
                        if image_aspect_ratio == 'anyres':
                            num_patch_width, num_patch_height = get_anyres_image_grid_shape(image_sizes[image_idx], self.config.image_grid_pinpoints, self.get_vision_tower().config.image_size)
                            image_feature = image_feature.view(num_patch_height, num_patch_width, height, width, -1)
                        else:
                            raise NotImplementedError
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = image_feature.permute(4, 0, 2, 1, 3).contiguous()
                            image_feature = image_feature.flatten(1, 2).flatten(2, 3)
                            image_feature = unpad_image(image_feature, image_sizes[image_idx])
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[:, None, None].expand(*image_feature.shape[:-1], 1).to(image_feature.device)
                            ), dim=-1)
                            image_feature = image_feature.flatten(1, 2).transpose(0, 1)
                        else:
                            image_feature = image_feature.permute(0, 2, 1, 3, 4).contiguous()
                            image_feature = image_feature.flatten(0, 3)
                        image_feature = torch.cat((base_image_feature, image_feature), dim=0)
                    else:
                        image_feature = image_feature[0]
                        if 'unpad' in mm_patch_merge_type:
                            image_feature = torch.cat((
                                image_feature,
                                self.model.image_newline[None].to(image_feature.device)
                            ), dim=0)
                    new_image_features.append(image_feature)
                image_features = new_image_features
            else:
                raise ValueError(f"Unexpected mm_patch_merge_type: {self.config.mm_patch_merge_type}")
        else:
            image_features = self.encode_images(
                images, text_embeds=text_embeds, text_attention_mask=text_attention_mask
            )

        # TODO: image start / end is not implemented here to support pretraining.
        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        # Let's just add dummy tensors if they do not exist,
        # it is a headache to deal with None all the time.
        # But it is not ideal, and if you have a better idea,
        # please open an issue / submit a PR, thanks.
        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
            labels = torch.full_like(input_ids, IGNORE_INDEX)

        # remove the padding using attention_mask -- FIXME
        _input_ids = input_ids
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        new_input_embeds = []
        new_labels = []
        cur_image_idx = 0
        for batch_idx, cur_input_ids in enumerate(input_ids):
            num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
            if num_images == 0:
                cur_image_features = image_features[cur_image_idx]
                cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                new_input_embeds.append(cur_input_embeds)
                new_labels.append(labels[batch_idx])
                cur_image_idx += 1
                continue

            image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
            cur_input_ids_noim = []
            cur_labels = labels[batch_idx]
            cur_labels_noim = []
            for i in range(len(image_token_indices) - 1):
                cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
            split_sizes = [x.shape[0] for x in cur_labels_noim]
            cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
            cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)
            cur_new_input_embeds = []
            cur_new_labels = []

            for i in range(num_images + 1):
                cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                cur_new_labels.append(cur_labels_noim[i])
                if i < num_images:
                    cur_image_features = image_features[cur_image_idx]
                    cur_image_idx += 1
                    cur_new_input_embeds.append(cur_image_features)
                    cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))

            cur_new_input_embeds = [x.to(self.device) for x in cur_new_input_embeds]

            cur_new_input_embeds = torch.cat(cur_new_input_embeds)
            cur_new_labels = torch.cat(cur_new_labels)

            new_input_embeds.append(cur_new_input_embeds)
            new_labels.append(cur_new_labels)

        # Truncate sequences to max length as image embeddings can make the sequence longer
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Combine them
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        for i, (cur_new_embed, cur_new_labels) in enumerate(zip(new_input_embeds, new_labels)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)

        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels

    def initialize_vision_tokenizer(self, model_args, tokenizer):
        if model_args.mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

        if model_args.mm_use_im_start_end:
            num_new_tokens = tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
            self.resize_token_embeddings(len(tokenizer))

            if num_new_tokens > 0:
                input_embeddings = self.get_input_embeddings().weight.data
                output_embeddings = self.get_output_embeddings().weight.data

                input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)
                output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
                    dim=0, keepdim=True)

                input_embeddings[-num_new_tokens:] = input_embeddings_avg
                output_embeddings[-num_new_tokens:] = output_embeddings_avg

            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = True
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False

            if model_args.pretrain_mm_mlp_adapter:
                mm_projector_weights = torch.load(model_args.pretrain_mm_mlp_adapter, map_location='cpu')
                embed_tokens_weight = mm_projector_weights['model.embed_tokens.weight']
                assert num_new_tokens == 2
                if input_embeddings.shape == embed_tokens_weight.shape:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight[-num_new_tokens:]
                elif embed_tokens_weight.shape[0] == num_new_tokens:
                    input_embeddings[-num_new_tokens:] = embed_tokens_weight
                else:
                    raise ValueError(f"Unexpected embed_tokens_weight shape. Pretrained: {embed_tokens_weight.shape}. Current: {input_embeddings.shape}. Numer of new tokens: {num_new_tokens}.")
        elif model_args.mm_use_im_patch_token:
            if model_args.tune_mm_mlp_adapter:
                for p in self.get_input_embeddings().parameters():
                    p.requires_grad = False
                for p in self.get_output_embeddings().parameters():
                    p.requires_grad = False
