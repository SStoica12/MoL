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

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

class LlavaConfig(LlamaConfig):
    model_type = "llava_llama"
    encoder_type: str = "clip" # or dino or clip_dino or c4


class LlavaLlamaModel(LlavaMetaModel, LlamaModel):
    config_class = LlavaConfig

    def __init__(self, config: LlamaConfig):
        # import pdb; pdb.set_trace()
        super(LlavaLlamaModel, self).__init__(config)


class LlavaLlamaForCausalLM(LlamaForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaConfig

    def __init__(self, config):
        # print("config: ", config)
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
        dino_images: Optional[torch.FloatTensor] = None, # if we add dino
        image_sizes: Optional[List[List[int]]] = None,
        clip_texts: Optional[List[str]] = None,
        return_dict: Optional[bool] = None,
        encoder_type="clip",
        fusion_type=False,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        # print("==================== I entered this forward ====================")
        # import sys; sys.exit(0)
        # import pdb; pdb.set_trace()
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
        dino_images: Optional[torch.Tensor] = None,
        image_sizes: Optional[torch.Tensor] = None,
        encoder_type="clip",
        prompts: Optional[List[str]] = None,
        **kwargs,
    ) -> Union[GenerateOutput, torch.LongTensor]:
        position_ids = kwargs.pop("position_ids", None)
        attention_mask = kwargs.pop("attention_mask", None)
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

class DoubleRouterLlavaLlamaForCausalLM(LlavaLlamaForCausalLM): # for both dino and clip
    def __init__(self, config, router = None, load_balance_alpha=10**-2, type="llava", aggregate_type = "attn"): # aggregate type will be one of linear, pool, attention. 
                                                                                                                # linear means to concat then pass through linear layer, 
                                                                                                                #  pool means to do line dino txt, and attention is 
                                                                                                                # Jeonghwan's idea
        # if we combine the features, we interleave clip and dino as input to the router. 
        super().__init__(config)
        self.router_clip = router
        self.router_dino = router
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
        self.rank0_print(" \n ================= I AM IN ROUTER ENCODE DINO IMAGES =================")
        vision_tower = self.get_model().get_dino_vision_tower()
        device = next(vision_tower.parameters()).device
        dtype = next(vision_tower.parameters()).dtype
        self.text_embeds_dino = text_embeds
        self.text_attention_mask_dino = text_attention_mask
        
        if type(dino_images) is list:
            image_features = []
            for image in dino_images:
                _, hidden_states = vision_tower(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True)
                image_features.append(hidden_states)
            hidden_states = torch.stack([hs for hs in image_features])
        else:
            _,hidden_states = vision_tower(dino_images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            hidden_states = torch.stack(list(hidden_states))
        assert hidden_states.ndim == 4
        image_features = self.router_dino(hidden_states, text_embeds=text_embeds, text_attention_mask=text_attention_mask, type="dino")
        assert image_features.requires_grad == True
        image_features = self.get_model().dino_mm_projector(image_features)
        assert image_features.requires_grad == True
        return image_features

    def encode_images_withclip(self, images, text_embeds=None, text_attention_mask=None, encoder_type="clip"):
        self.rank0_print(f" \n ================== I AM IN ROUTER ENCODE IMAGES WITH CLIP using encoder {encoder_type}! ==================")
        # import pdb; pdb.set_trace()
        vision_tower = self.get_model().get_vision_tower(load_model=encoder_type).vision_tower
        device = vision_tower.device
        dtype = vision_tower.dtype
        self.text_embeds_clip = text_embeds
        self.text_attention_mask_clip = text_attention_mask
        
        if type(images) is list:
            image_features = []
            for image in images:
                image_features.append(vision_tower(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True).hidden_states)
            hidden_states = torch.stack([hs for hs in image_features])
        else:
            image_forward_outs = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            hidden_states = torch.stack(list(image_forward_outs.hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        
        assert hidden_states.ndim == 4
        image_features = self.router_clip(hidden_states[1:, :, :, :], text_embeds=text_embeds, text_attention_mask=text_attention_mask, type=encoder_type) # do not want to include the embedding layer
        assert image_features.requires_grad == True
        # import pdb; pdb.set_trace()
        image_features = self.get_model().mm_projector(image_features) # Just one project?
        # assert image_features.requires_grad == True
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
        import pdb; pdb.set_trace()

        load_balanace_loss_clip = None
        load_blanace_loss_dino = None
        cached_router_probs_clip = getattr(self.router_clip, "_latest_router_probs_for_aux_clip", None)
        cached_router_probs_dino = getattr(self.router_dino, "_latest_router_probs_for_aux_dino", None)
        if cached_router_probs_clip is not None:
            load_balanace_loss_clip = self.router_clip.load_balance_loss(cached_router_probs_clip)
        if cached_router_probs_dino:
            load_blanace_loss_dino =  self.router_dino.load_balance_loss(cached_router_probs_dino)

        if load_balanace_loss_clip is not None and load_blanace_loss_dino is not None:
            return load_balanace_loss_clip + load_blanace_loss_dino #TODO: change if needed

        vision_tower_clip = self.get_model().get_vision_tower().vision_tower
        if hasattr(self.get_model(), "get_dino_vision_tower"): # this works fine for seperate instances but not for when I have both. TODO: why?
            vision_tower_dino = self.get_model().get_dino_vision_tower().vision_tower
        else:
            vision_tower_dino = self.get_model().get_vision_tower(load_model="dino")
        device = vision_tower_clip.device
        dtype = vision_tower_clip.dtype
        losses = []
        if type(images) is list:
            losses_clip = []
            losses_dino = []
            for image, dino_image in zip(images, dino_images):
                image_forward_outs_clip = vision_tower_clip(image.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
                image_forward_outs_dino = vision_tower_dino(dino_image.to(device=device, dtype=dtype), output_hidden_states=True)
                router_probs_clip = self.get_router_probs_clip(image_forward_outs_clip.hidden_states)
                router_probs_dino = self.get_router_probs_dino(image_forward_outs_dino.hidden_states)
                losses_clip.append(self.router_clip.load_balance_loss(router_probs_clip))
                losses_dino.append(self.router_dino.load_balance_loss(router_probs_dino))
            loss = sum(losses_clip) / len(losses_clip) + sum(losses_dino) / len(losses_dino)
        else:
            image_forward_outs_clip = vision_tower_clip(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            image_forward_outs_dino, hidden_states = vision_tower_dino(dino_images.to(device=device, dtype=dtype), output_hidden_states=True)
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
        # import pdb; pdb.set_trace()
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
        # print("hidden_states : ", hidden_states.device)
        # print("self.router_dino parameters", next(self.router_dino.parameters()).device)
        # import pdb; pdb.set_trace()
        image_features = self.router_dino(hidden_states, text_embeds=text_embeds, text_attention_mask=text_attention_mask, type="dino") # these are currently on cuda: 0
        # assert image_features.requires_grad == True
        image_features = self.get_model().dino_mm_projector(image_features)
        # assert image_features.requires_grad == True
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
        # import pdb; pdb.set_trace()
        image_features = self.router_clip(hidden_states[1:, :, :, :], text_embeds=text_embeds, text_attention_mask=text_attention_mask, type=type_encoder) # do not want to include the embedding layer
        assert image_features.requires_grad == True
        # import pdb; pdb.set_trace()
        image_features = self.get_model().mm_projector(image_features) # Just one project?
        # assert image_features.requires_grad == True
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
        # import pdb; pdb.set_trace()
        cached_router_probs_clip = None
        cached_router_probs_dino = None
        if "clip" in self.encoder_type or "siglip" in self.encoder_type:
            cached_router_probs_clip = getattr(self.router_clip, "_latest_router_probs_for_aux_clip", None)
        if "dino" in self.encoder_type:
            cached_router_probs_dino = getattr(self.router_dino, "_latest_router_probs_for_aux_dino", None)
        
        if cached_router_probs_clip is not None:
            loss = self.router_clip.load_balance_loss(cached_router_probs_clip)
            rank0_print(f"Using cached router probs for clip, load balance loss: {loss.item()}")
            return loss
        if cached_router_probs_dino is not None:
            loss = self.router_dino.load_balance_loss(cached_router_probs_dino)
            rank0_print(f"Using cached router probs for dino, load balance loss: {loss.item()}")
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
            # import pdb; pdb.set_trace()
            if "dino" in self.encoder_type:
                # import pdb; pdb.set_trace()
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
