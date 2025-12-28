import pdb
import torch
import torch.nn as nn
from typing import Optional
from transformers import CLIPVisionModel, CLIPImageProcessor, CLIPVisionConfig
import sys
import typing
from typing import Any, TypeAlias, TypedDict, Union
import typing_extensions

# class ScaledResidualEncoderLayer(nn.Module):
#    def __init__(self, layer):
#        super().__init__()
#        self.scaling_coef = nn.Parameter(torch.ones(1), requires_grad=True)
#        self.layer = layer
#    def forward(self, hidden_states: torch.Tensor, attention_mask: torch.Tensor, causal_attention_mask: torch.Tensor, output_attentions: Optional[bool] = False):
#        output = self.layer(hidden_states, attention_mask, causal_attention_mask, output_attentions)
#        # Scale the hidden states (output[0]) while preserving the output structure
#        scaled_hidden_states = output[0] * self.scaling_coef
#        if output_attentions and len(output) > 1:
#            return (scaled_hidden_states,) + output[1:]
#        else:
#            return (scaled_hidden_states,)

class TransformersKwargs(TypedDict, total=False):
    """
    Keyword arguments to be passed to the forward pass of a `PreTrainedModel`.

    Attributes:
        num_items_in_batch (`Optional[torch.Tensor]`, *optional*):
            Number of items in the batch. It is recommended to pass it when you are doing gradient accumulation.
        output_hidden_states (`Optional[bool]`, *optional*):
            Most of the models support outputting all hidden states computed during the forward pass.
        output_attentions (`Optional[bool]`, *optional*):
            Turn this on to return the intermediary attention scores.
        output_router_logits (`Optional[bool]`, *optional*):
            For MoE models, this allows returning the router logits to compute the loss.
        cu_seq_lens_q (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for query state.
        cu_seq_lens_k (`torch.LongTensor`, *optional*)
            Gets cumulative sequence length for key state.
        max_length_q (`int`, *optional*):
            Maximum sequence length for query state.
        max_length_k (`int`, *optional*):
            Maximum sequence length for key state.
        position_ids (`torch.LongTensor`, *optional*)
            Indices of positions of each input sequence tokens.
    """

    num_items_in_batch: Optional["torch.Tensor"]
    output_hidden_states: bool | None
    output_attentions: bool | None
    output_router_logits: bool | None
    cu_seq_lens_q: Optional["torch.LongTensor"]
    cu_seq_lens_k: Optional["torch.LongTensor"]
    max_length_q: int | None
    max_length_k: int | None
    position_ids: Optional["torch.LongTensor"]

if sys.version_info >= (3, 11):
    Unpack = typing.Unpack
else:
    Unpack = typing_extensions.Unpack

class ScaledResidualEncoderLayer(nn.Module):
    def __init__(self, layer):
        super().__init__()
        self.scaling_coef = nn.Parameter(torch.ones(2), requires_grad=True)
        self.layer = layer
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        causal_attention_mask: torch.Tensor, 
        output_attentions: Optional[bool] = False,
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.FloatTensor:
        residual = hidden_states
        # print("===== entered ===== ")
        hidden_states = self.layer.layer_norm1(hidden_states)
        # print("===== done with layer norm 1 ===== ")
        hidden_states, _ = self.layer.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            **kwargs,
        )
        # print("===== done with attention ===== ")
        hidden_states = residual + self.scaling_coef[0] * hidden_states
        # import pdb; pdb.set_trace()
        # print("===== done with residual scaling 1 ===== ")
        residual = hidden_states
        hidden_states = self.layer.layer_norm2(hidden_states)
        # print("===== done with layer norm 2 ===== ")
        hidden_states = self.layer.mlp(hidden_states)
        # print("===== done with mlp ===== ")
        hidden_states = residual + self.scaling_coef[1] * hidden_states
        # print("===== done with residual scaling 2 ===== ")
        # print("hidden_states.shape: ", hidden_states.shape )
        return (hidden_states, )


class CLIPVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = CLIPVisionConfig.from_pretrained(self.vision_tower_name)

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        
        # Replace layers BEFORE freezing to ensure new parameters are properly registered
        layers = self.vision_tower.vision_model.encoder.layers
        for i in range(len(layers)):
            layer = layers[i]
            self.vision_tower.vision_model.encoder.layers[i] = ScaledResidualEncoderLayer(layer)

        # Now freeze the vision_tower (this will freeze all parameters)
        self.vision_tower.requires_grad_(False)

        # Then explicitly enable gradients for scaling_coef parameters
        for n, p in self.vision_tower.named_parameters():
            if "scaling_coef" in n:
                p.requires_grad = True

        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        image_features = image_forward_outs.hidden_states[self.select_layer]
        if self.select_feature == 'patch':
            image_features = image_features[:, 1:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        # pdb.set_trace()
        # print("grad_enabled in CLIPVisionTower.forward =", torch.is_grad_enabled())
        if type(images) is list:
            image_features = []
            # x = images.to(device=self.device, dtype=self.dtype)
            # x = x.detach().requires_grad_(True)  # <-- only makes IMAGE tensor require grad
            for image in images:
                image_forward_out = self.vision_tower(image.to(device=self.device, dtype=self.dtype).unsqueeze(0), output_hidden_states=True)
                image_feature = self.feature_select(image_forward_out).to(image.dtype)
                image_features.append(image_feature)
        else:
            image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
            image_features = self.feature_select(image_forward_outs).to(images.dtype)
        # pdb.set_trace() # placed pdb here to find where the project was getting called -- press u multiple times
        return image_features

    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return self.vision_tower.dtype

    @property
    def device(self):
        return self.vision_tower.device

    @property
    def config(self):
        if self.is_loaded:
            return self.vision_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        return self.config.hidden_size

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size

    @property
    def num_patches(self):
        return (self.config.image_size // self.config.patch_size) ** 2



class CLIPVisionTowerS2(CLIPVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__(vision_tower, args, delay_load)

        self.s2_scales = getattr(args, 's2_scales', '336,672,1008')
        self.s2_scales = list(map(int, self.s2_scales.split(',')))
        self.s2_scales.sort()
        self.s2_split_size = self.s2_scales[0]
        self.s2_image_size = self.s2_scales[-1]

        try:
            from s2wrapper import forward as multiscale_forward
        except ImportError:
            raise ImportError('Package s2wrapper not found! Please install by running: \npip install git+https://github.com/bfshi/scaling_on_scales.git')
        self.multiscale_forward = multiscale_forward

        # change resize/crop size in preprocessing to the largest image size in s2_scale
        if not delay_load or getattr(args, 'unfreeze_mm_vision_tower', False):
            self.image_processor.size['shortest_edge'] = self.s2_image_size
            self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = CLIPImageProcessor.from_pretrained(self.vision_tower_name)
        self.vision_tower = CLIPVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        
        # Replace layers BEFORE freezing to ensure new parameters are properly registered
        layers = self.vision_tower.vision_model.encoder.layers
        for i in range(len(layers)):
            layer = layers[i]
            self.vision_tower.vision_model.encoder.layers[i] = ScaledResidualEncoderLayer(layer)

        # Now freeze the vision_tower (this will freeze all parameters)
        self.vision_tower.requires_grad_(False)

        # Then explicitly enable gradients for scaling_coef parameters
        for n, p in self.vision_tower.named_parameters():
            if "scaling_coef" in n:
                p.requires_grad = True

        self.image_processor.size['shortest_edge'] = self.s2_image_size
        self.image_processor.crop_size['height'] = self.image_processor.crop_size['width'] = self.s2_image_size

        self.is_loaded = True

    # @torch.no_grad()
    def forward_feature(self, images):
        image_forward_outs = self.vision_tower(images.to(device=self.device, dtype=self.dtype), output_hidden_states=True)
        image_features = self.feature_select(image_forward_outs).to(images.dtype)
        return image_features

    # @torch.no_grad()
    def forward(self, images):
        if type(images) is list:
            image_features = []
            for image in images:
                image_feature = self.multiscale_forward(self.forward_feature, image.unsqueeze(0), img_sizes=self.s2_scales, max_split_size=self.s2_split_size)
                image_features.append(image_feature)
        else:
            image_features = self.multiscale_forward(self.forward_feature, images, img_sizes=self.s2_scales, max_split_size=self.s2_split_size)

        return image_features

    @property
    def hidden_size(self):
        return self.config.hidden_size * len(self.s2_scales)
