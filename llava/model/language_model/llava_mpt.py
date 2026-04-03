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


from typing import Optional, Tuple, List

import torch

import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM, \
                         MptConfig, MptForCausalLM, MptModel
from llava.model.llava_arch import LlavaMetaModel, LlavaMetaForCausalLM


class LlavaMptConfig(MptConfig):
    model_type = "llava_mpt"


class LlavaMptModel(LlavaMetaModel, MptModel):
    config_class = LlavaMptConfig

    def __init__(self, config: MptConfig):
        config.hidden_size = config.d_model
        super(LlavaMptModel, self).__init__(config)
    
    def embed_tokens(self, x):
        return self.wte(x)


class LlavaMptForCausalLM(MptForCausalLM, LlavaMetaForCausalLM):
    config_class = LlavaMptConfig
    supports_gradient_checkpointing = True

    def __init__(self, config):
        super(MptForCausalLM, self).__init__(config)

        self.transformer = LlavaMptModel(config)
        self.lm_head = torch.nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def get_model(self):
        return self.transformer

    def _set_gradient_checkpointing(self, module, value=False):
        if isinstance(module, LlavaMptModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.Tensor, torch.Tensor], ...]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        images=None,
        clip_texts: Optional[List[str]] = None):

        input_ids, position_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(
            input_ids=input_ids,
            position_ids=None,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            labels=labels,
            images=images,
            image_sizes=None,
            clip_texts=clip_texts,
        )
        
        return super().forward(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None, inputs_embeds=None, **kwargs):
        images = kwargs.pop("images", None)
        _inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values, inputs_embeds=inputs_embeds, **kwargs
        )
        _inputs['images'] = images
        return _inputs

class RouterLlavaMptForCausalLM(LlavaMptForCausalLM):
    def __init__(self, config, router = None, load_balance_alpha=10**-2, type="llava"):
        super().__init__(config)
        self.router = router
        self.load_balance_alpha = load_balance_alpha
    def get_model(self):
        return self.model
    def encode_images(self, images, text_embeds=None, text_attention_mask=None):
        print("--------- I AM IN ROUTER ENCODE IMAGES ---------")
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
        assert image_features.requires_grad == True
        # import pdb; pdb.set_trace()
        image_features = self.get_model().mm_projector(image_features) # image_features.requires_grad = True
        # assert image_features.requires_grad == True
        return image_features
    
    def get_router_probs(self, hidden_states):
        hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        router_probs = self.router.router_probs(hidden_states[1:, :, :, :]) # remove embedding
        return router_probs
    
    def get_aux_loss(self, images):
        # calculate the hidden states
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

class DoubleRouterLlavaMptForCausalLM(LlavaMptForCausalLM): # for both dino and clip
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
    
    def get_model(self):
        return self.model
    
    def encode_images(self, images):
        print("--------- I AM IN DOUBLE ROUTER ENCODE IMAGES ---------")
        vision_tower_dino = self.get_model().get_dino_vision_tower().vision_tower
        vision_tower_clip = self.get_model().get_vision_tower().vision_tower
        device = vision_tower_clip.device
        dtype = vision_tower_clip.dtype

        if vision_tower_clip.device != vision_tower_dino.device:
            vision_tower_dino = vision_tower_dino.to(device=device, dtype=dtype)
        
        image_features_clip = []
        image_features_dino = []
        if type(images) is list:
            for image in images:
                image_features_clip.append(vision_tower_clip(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True))
                image_features_dino.append(vision_tower_dino(image.to(device=device, dtype=dtype).unsqueeze(0), output_hidden_states=True))
        else:
            image_features_clip.append(vision_tower_clip(images.to(device=device, dtype=dtype), output_hidden_states=True))
            image_features_dino.append(vision_tower_dino(images.to(device=device, dtype=dtype), output_hidden_states=True)) # all 25 layers have  torch.Size([32, 577, 1024])
        
        hidden_states_clip = image_features_clip.stack(dim=0) 
        hidden_stapes_dino = image_features_dino.stack(dim=0)
        #hidden_states = torch.stack(list(image_forward_outs.hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        assert hidden_states_clip.ndim == 4 # may not pass now
        assert hidden_stapes_dino.ndim == 4
        image_features_clip = self.router(hidden_states_clip[1:, :, :, :]) # do not want to include the embedding layer
        image_features_dino = self.router(hidden_stapes_dino[1:, :, :, :])

        # assert image_features_clip.requires_grad == True
        # import pdb; pdb.set_trace()
        image_features_clip = self.get_model().mm_projector(image_features_clip) # image_features.requires_grad = True
        image_features_dino = self.get_model().dino_mm_projector(image_features_dino)

        """
        Now the question is what to do with the features
        1. Concatenate the features and then pass through linear layers like one of Jeonghwan's ideas
        2. We can do something like dino txt and pool the dino text features and concatenate with clip features. I think this results in a problem with shapes, so may avoid.
        do we want to do a weighted average? <- research question? @TODO: if we want to do this we need to do a text conditioning on alpha --> alphas * clip + (1 - alphas) * dino
        3. Clip as Q and Dino as KV. So cross attention through clip and dino --> semantically agregation of Clip and Dino. Enhance Dino
        """
        if self.aggregate_type == "linear":
            image_features = torch.cat([image_features_clip, image_features_dino], dim=-1) # check shapes --- if not the same shape, then we need MLP or just ignore.
            image_features = self.linear(image_features)
        
        # elif self.aggregate_type == "pool":
        #     # Try and do as close to dino txt ad ave g = [c', average_pool(dino patches)]: 
        #     # https://github.com/facebookresearch/dinov2/blob/b194f00db6136677fc8a4cc2ef2168f7699dfba2/dinov2/eval/linear.py#L171
        #     image_features = torch.cat(
        #         (
        #             image_features,
        #             torch.mean(image_features_dino, dim=1),  # patch tokens
        #         ),
        #         dim=-1,
        #     )
        #     image_features = image_features.reshape(image_features.shape[0], -1) # CHECK. 
        #     image_features = torch.cat([image_features, image_features_clip], dim=-1)
        
        else:
            Q = image_features_clip
            K = image_features_dino
            V = image_features_dino
            attn_weights = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
            attn_weights = torch.softmax(attn_weights, dim=-1)
            image_features = torch.matmul(attn_weights, V)
        # assert image_features.requires_grad == True
        return image_features
    
    def get_router_prob_clip(self, hidden_states):
        hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        router_probs = self.router_clip.router_probs(hidden_states[1:, :, :, :]) # remove embedding
        return router_probs

    def get_router_prob_dino(self, hidden_states):
        hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        router_probs = self.router_dino.router_probs(hidden_states[1:, :, :, :]) # remove embedding
        return router_probs
    
    def get_aux_loss(self, images):
        # calculate the hidden states
        vision_tower_clip = self.get_model().get_vision_tower().vision_tower
        vision_tower_dino = self.get_model().get_dino_vision_tower().vision_tower
        device = vision_tower_clip.device
        dtype = vision_tower_clip.dtype
        losses = []
        if type(images) is list:
            losses_clip = []
            losses_dino = []
            for image in images:
                image_forward_outs_clip = vision_tower_clip(image.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
                image_forward_outs_dino = vision_tower_dino(image.to(device=device, dtype=dtype), output_hidden_states=True)
                router_probs_clip = self.get_router_probs_clip(image_forward_outs_clip.hidden_states)
                router_probs_dino = self.get_router_probs_dino(image_forward_outs_dino.hidden_states)
                losses_clip.append(self.router_clip.load_balance_loss(router_probs_clip))
                losses_dino.append(self.router_dino.load_balance_loss(router_probs_dino))
                losses.append(sum(losses_clip) / len(losses_clip) + sum(losses_dino) / len(losses_dino)) # how do I make this better? Do I introduce scaling coefs?
            loss = sum(losses) / len(losses)
        else:
            image_forward_outs_clip = vision_tower_clip(image.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            image_forward_outs_dino = vision_tower_dino(image.to(device=device, dtype=dtype), output_hidden_states=True)
            router_probs_clip = self.get_router_probs_clip(image_forward_outs_clip.hidden_states)
            router_probs_dino = self.get_router_probs_dino(image_forward_outs_dino.hidden_states)
            loss_clip = self.router_clip.load_balance_loss(router_probs_clip)
            loss_dino = self.router_dino.load_balance_loss(router_probs_dino)
            loss = loss_clip + loss_dino # how do I make this better? Do I introduce scaling coefs?
        return loss

AutoConfig.register("llava_mpt", LlavaMptConfig)
AutoModelForCausalLM.register(LlavaMptConfig, LlavaMptForCausalLM, RouterLlavaMptForCausalLM)
AutoModelForCausalLM.register(LlavaMptConfig, DoubleRouterLlavaMptForCausalLM)