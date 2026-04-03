import torch
import torch.nn as nn
import re


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, type="clip", **kwargs):
    """
    We need to change the input to the MLP if we have dino from 5120 to 1024
    (or whatever the specific dino implementation uses).
    """
    # import pdb; pdb.set_trace()
    fusion_type = config.fusion_type
    projector_type = getattr(config, 'mm_projector_type', 'linear')
    if type == "dino":
        input = getattr(config, 'mm_hidden_size_dino', 1024)
    else:
        input = getattr(config, 'mm_hidden_size', 1024)
    
    if projector_type == 'linear':
        return nn.Linear(input, config.mm_hidden_size)

    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(input, config.hidden_size)]
        for i in range(1, mlp_depth):
            modules.append(nn.GELU())
            if i == mlp_depth - 1 and fusion_type == "concate_upsample":
                modules.append(nn.Linear(config.hidden_size, config.hidden_size * 2))
            else:
                modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        if type == "dino":
            raise ValueError('Identity projector is not supported for DINO vision tower since the feature dimension does not match the language tower. Please us linear if you do not want to use the 2 layer MLP')
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')
