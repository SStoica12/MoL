import pdb
import torch
import torch.nn as nn
from typing import Optional
from transformers import SiglipVisionConfig, SiglipImageProcessor, SiglipVisionModel, SiglipTokenizer, SiglipTextModel
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
        ref_param = next(layer.parameters(), None)
        if ref_param is None:
            self.scaling_coef = nn.Parameter(torch.ones(2), requires_grad=True)
        else:
            self.scaling_coef = nn.Parameter(
                torch.ones(2, device=ref_param.device, dtype=ref_param.dtype),
                requires_grad=True,
            )
        self.layer = layer
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        causal_attention_mask: Optional[torch.Tensor] = None,
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
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(device=hidden_states.device, dtype=hidden_states.dtype)
        scaling_coef_0 = self.scaling_coef[0]
        if scaling_coef_0.device != hidden_states.device or scaling_coef_0.dtype != hidden_states.dtype:
            scaling_coef_0 = scaling_coef_0.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = residual + scaling_coef_0 * hidden_states
        # import pdb; pdb.set_trace()
        # print("===== done with residual scaling 1 ===== ")
        residual = hidden_states
        hidden_states = self.layer.layer_norm2(hidden_states)
        # print("===== done with layer norm 2 ===== ")
        hidden_states = self.layer.mlp(hidden_states)
        # print("===== done with mlp ===== ")
        if residual.device != hidden_states.device or residual.dtype != hidden_states.dtype:
            residual = residual.to(device=hidden_states.device, dtype=hidden_states.dtype)
        scaling_coef_1 = self.scaling_coef[1]
        if scaling_coef_1.device != hidden_states.device or scaling_coef_1.dtype != hidden_states.dtype:
            scaling_coef_1 = scaling_coef_1.to(device=hidden_states.device, dtype=hidden_states.dtype)
        hidden_states = residual + scaling_coef_1 * hidden_states
        # print("===== done with residual scaling 2 ===== ")
        # print("hidden_states.shape: ", hidden_states.shape )
        return (hidden_states, )

class SiglipVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False

        self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        self.vision_tower_delta = getattr(args, "vision_tower_delta", False)
        self._vision_tower_deltas = None
        self._delta_hook_handles = []

        if not delay_load:
            self.load_model()
        elif getattr(args, 'unfreeze_mm_vision_tower', False):
            self.load_model()
        else:
            self.cfg_only = SiglipVisionConfig.from_pretrained(self.vision_tower_name)

    def _get_layer_mlp(self, layer):
        if hasattr(layer, "mlp"):
            return layer.mlp
        # Compatibility for wrapped layers.
        if hasattr(layer, "layer") and hasattr(layer.layer, "mlp"):
            return layer.layer.mlp
        raise AttributeError("Unable to locate MLP module in vision encoder layer.")

    def _register_delta_hooks(self):
        if not self.vision_tower_delta or self._delta_hook_handles:
            return

        layers = self.vision_tower.vision_model.encoder.layers
        num_layers = len(layers)

        def _reset_delta_cache(_module, _inputs):
            self._vision_tower_deltas = [None] * num_layers

        def _capture_delta(layer_idx):
            def _hook(_module, _inputs, output):
                if self._vision_tower_deltas is not None:
                    self._vision_tower_deltas[layer_idx] = output
            return _hook

        def _replace_hidden_states(hidden_states):
            if hidden_states is None or not isinstance(hidden_states, (tuple, list)):
                return hidden_states
            if self._vision_tower_deltas is None:
                return hidden_states
            if len(hidden_states) != len(self._vision_tower_deltas) + 1:
                return hidden_states
            if any(delta is None for delta in self._vision_tower_deltas):
                raise RuntimeError(
                    "vision_tower_delta=True, but failed to capture all per-layer MLP deltas."
                )
            return (hidden_states[0], *self._vision_tower_deltas)  # [x0, delta1, delta2, ..., deltaL]

        def _finalize_output(_module, _inputs, output):
            if isinstance(output, tuple):
                if len(output) >= 3:
                    output = list(output)
                    output[2] = _replace_hidden_states(output[2])
                    output = tuple(output)
                self._vision_tower_deltas = None
                return output

            if hasattr(output, "hidden_states"):
                output.hidden_states = _replace_hidden_states(output.hidden_states)
            self._vision_tower_deltas = None
            return output

        self._delta_hook_handles.append(self.vision_tower.register_forward_pre_hook(_reset_delta_cache))
        for i, layer in enumerate(layers):
            mlp = self._get_layer_mlp(layer)
            self._delta_hook_handles.append(mlp.register_forward_hook(_capture_delta(i)))
        self._delta_hook_handles.append(self.vision_tower.register_forward_hook(_finalize_output))

    def load_model(self, device_map=None):
        if self.is_loaded:
            print('{} is already loaded, `load_model` called again, skipping.'.format(self.vision_tower_name))
            return

        self.image_processor = SiglipImageProcessor.from_pretrained(self.vision_tower_name)
        self.image_processor.crop_size = self.image_processor.size # added as bunny did
        self.vision_tower = SiglipVisionModel.from_pretrained(self.vision_tower_name, device_map=device_map)
        self._register_delta_hooks()

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

class SiglipVisionTextTower(SiglipVisionTower):
    def __init__(self, vision_tower, args, delay_load=False):
        self.text_tower = None
        self.text_tokenizer = None
        super().__init__(vision_tower, args, delay_load)

    def load_model(self, device_map=None):
        super().load_model(device_map=device_map)
        if self.text_tower is None:
            # import pdb; pdb.set_trace()
            self.text_tokenizer = SiglipTokenizer.from_pretrained(self.vision_tower_name)
            self.text_tower = SiglipTextModel.from_pretrained(self.vision_tower_name, device_map=device_map)
            self.text_tower.requires_grad_(False)

    def get_text_tower(self):  # NOTE: this will be used to return a handler to self.text_tower
        if self.text_tower is None:
            raise RuntimeError("Text tower not loaded. Please call load_model() first.")
        return self.text_tower

    def encode_text(self, input_ids, attention_mask=None):
        text_out = self.get_text_tower()(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        return text_out.last_hidden_state
