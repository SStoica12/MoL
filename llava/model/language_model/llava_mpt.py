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


from typing import Optional, Tuple

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
        images=None):

        input_ids, attention_mask, past_key_values, inputs_embeds, labels = self.prepare_inputs_labels_for_multimodal(input_ids, attention_mask, past_key_values, labels, images)
        
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
    def encode_images(self, images):
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
        image_features = self.router(hidden_states[1:, :, :, :]) # do not want to include the embedding layer
        assert image_features.requires_grad == True
        # import pdb; pdb.set_trace()
        image_features = self.get_model().mm_projector(image_features) # image_features.requires_grad = True
        # assert image_features.requires_grad == True
        return image_features
    
    def get_router_probs_and_expert_mask(self, hidden_states):
        hidden_states = torch.stack(list(hidden_states)) # [num_layers, batch_size, num_image_tokens, vision_hidden_dim]
        router_probs = self.router.router_probs(hidden_states[1:, :, :, :]) # remove embedding
        _, _, _, expert_mask = self.router.get_layer_and_scaling_probs_and_expert_mask(router_probs)
        return router_probs, expert_mask
    
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
                router_probs, expert_mask = self.get_router_probs_and_expert_mask(image_forward_outs.hidden_states)
                losses.append(self.router.load_balance_loss(router_probs, expert_mask))
            loss = sum(losses) / len(losses)
        else:
            image_forward_outs = vision_tower(images.to(device=device, dtype=dtype), output_hidden_states=True) # all 25 layers have  torch.Size([32, 577, 1024])
            router_probs, expert_mask = self.get_router_probs_and_expert_mask(image_forward_outs.hidden_states)
            loss = self.router.load_balance_loss(router_probs, expert_mask)
        return loss

AutoConfig.register("llava_mpt", LlavaMptConfig)
AutoModelForCausalLM.register(LlavaMptConfig, LlavaMptForCausalLM, RouterLlavaMptForCausalLM)
