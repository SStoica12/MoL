import torch
import torch.nn as nn
import torchvision.transforms as T
from transformers import CLIPVisionConfig, CLIPTokenizer, CLIPTextModel
import torch.nn.functional as F
from types import SimpleNamespace

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

def rank0_print(*args):
    import os
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if local_rank == 0:
        print(*args)

class DINOVisionConfig(CLIPVisionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_act = 'gelu'
        self.mm_hidden_size = 1024
        self.image_size = 224
        self.layer_norm_eps = 1e-6
        self.model_type = 'dinov2'
        self.projection_dim = 1024
        self.num_register_tokens = 0
        self.patch_size = 14  # taken from dino config. Clip has 32
        self.hidden_size_dino = 1024


class DINOTxTVisionConfig(CLIPVisionConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hidden_act = 'gelu'
        self.mm_hidden_size = 1024
        self.image_size = 224
        self.layer_norm_eps = 1e-6
        self.model_type = 'dinov2'
        # self.projection_dim = 2048    # concat([CLS], patch_avg) = 1024 + 1024
        self.num_register_tokens = 4  # was a local var before, must be self.
        self.patch_size = 14
        self.hidden_size_dino = 1024

class DINOVisionImageProcessor(T.Resize):
    def __init__(self, image_size=224):
        super().__init__(size=image_size, interpolation=T.InterpolationMode.BICUBIC)
        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if hasattr(img, 'mode') and img.mode != 'RGB' else img),
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),  # use param, not hardcoded 224
            T.ToTensor(),
            T.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ])
        self.image_mean = IMAGENET_DEFAULT_MEAN
        self.image_std = IMAGENET_DEFAULT_STD
        self.crop_size = image_size

    def preprocess(self, image,  return_tensors='pt'):
        # follow the expected of processor if we used CLip.
        res = {"pixel_values" : [self.transform(image)]}
        return res

class DINOVisionTower(nn.Module):
    def __init__(self, vision_tower, args, delay_load=False):
        super().__init__()

        self.is_loaded = False
        if "clip" in vision_tower: # happens if we have clip + dino instance
            idx = vision_tower.find("dino")
            self.vision_tower_name = vision_tower[idx:]
        else:
            self.vision_tower_name = vision_tower
        self.select_layer = args.mm_vision_select_layer
        self.select_feature = getattr(args, 'mm_vision_select_feature', 'patch')
        # import pdb; pdb.set_trace()
        if not delay_load:
            self.load_model()
        else:
            if self.vision_tower_name == "dinov2_txt":
                self.config = DINOTxTVisionConfig()
            else:
                self.config = DINOVisionConfig()

    def load_model(self, device_map=None):
        # import pdb; pdb.set_trace()
        if self.vision_tower_name == "dinov2_base": # done. Seems to train ok?
            self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
            self.config = DINOVisionConfig()
        elif self.vision_tower_name == "dinov2_registers":
            self.vision_tower = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg')
            self.config = DINOVisionConfig()
            self.config.num_register_tokens = 4
        elif self.vision_tower_name == "dinov2_txt":
            self.vision_tower = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l'
            ).visual_model
            self.config = DINOTxTVisionConfig()
        else:
            raise ValueError(f'Unknown vision tower: {self.vision_tower_name}, please choose among dinov2_base, dinov2_registers, and dinov2_txt')
        self.image_processor = DINOVisionImageProcessor(image_size=self.config.image_size)
        self.vision_tower.requires_grad_(False)
        self.is_loaded = True

    def feature_select(self, image_forward_outs):
        # get_class_and_patch_tokens already removes the patch tokens
        # import pdb; pdb.set_trace()
        image_features = image_forward_outs["x_prenorm"]
        if self.vision_tower_name == "dinov2_base":
            assert self.config.num_register_tokens == 0
        if self.select_feature == 'patch':
            if "txt" in self.vision_tower_name:
                image_features = image_features #[:, 1:]   # registers already stripped
            else:
                image_features = image_features[:, 1 + self.config.num_register_tokens:]
        elif self.select_feature == 'cls_patch':
            image_features = image_features  # CLS + patches already correct for txt
        else:
            raise ValueError(f'Unexpected select feature: {self.select_feature}')
        return image_features

    @torch.no_grad()
    def forward(self, images, output_hidden_states=False, output_attentions=False):
        # we need to wrap forward_features to get the hidden states.
        # import pdb; pdb.set_trace()
        # if output_hidden_states:
        # import pdb; pdb.set_trace()
        def _forward_features(model, imgs):
            # import pdb; pdb.set_trace()
            if not output_hidden_states:
                if "txt" in self.vision_tower_name:
                    # https://github.com/facebookresearch/dinov2/blob/7b187bd4df8efce2cbcbbb67bd01532c19bf4c9c/dinov2/hub/text/vision_tower.py#L162
                    device = next(model.parameters()).device # for some reason images were on device 2 rather than the device the model parameters were on
                    _, patch_tokens = model.get_class_and_patch_tokens(images.to(device=device)) # we can not do forward because it would pool the patch tokens, and llava projector expects all 257 patch tokens
                    return {"x_prenorm" : patch_tokens}, None, None # this should hopefully be ok?
                else:
                    return model.forward_features(imgs), None, None # forward_features is dict of dict_keys(['x_norm_clstoken', 'x_norm_regtokens', 'x_norm_patchtokens', 'x_prenorm', 'masks']). x_prenorm has shape torch.Size([32, 257 or 261, 1024])
            # import pdb; pdb.set_trace()
            hidden_states = []
            attentions = []
            if "registers" in self.vision_tower_name or "txt" in self.vision_tower_name: # we need to remove the registers
                if "registers" in self.vision_tower_name:
                    blocks = model.blocks
                    norm_layer = model.norm
                elif "txt" in self.vision_tower_name:
                    # we need to get the hidden states from the backbone, normalize them, and then feed them
                    # thorugh the heads.
                    blocks = model.backbone.model.blocks
                    norm_layer = model.backbone.model.norm
                def forward_hook(module, input, output):
                    output = norm_layer(output)
                    # import pdb; pdb.set_trace()
                    # output = output[:, 1+self.config.num_register_tokens:, :]
                    hidden_states.append(output)
                hooks = [
                    block.register_forward_hook(forward_hook)
                    for block in blocks
                ]
            else:
                norm_layer = model.norm
                blocks = model.blocks
                hooks = [
                    block.register_forward_hook(lambda module, input, output: hidden_states.append(norm_layer(output)))
                    for block in model.blocks # this had [:self.select_layer] ?
                ]
            attention_hooks = []
            if output_attentions:
                num_register_tokens = self.config.num_register_tokens

                def attention_hook(module, inputs):
                    attn_input = inputs[0]
                    B, N, C = attn_input.shape
                    qkv = module.qkv(attn_input).reshape(
                        B,
                        N,
                        3,
                        module.num_heads,
                        C // module.num_heads,
                    )
                    q, k, _ = torch.unbind(qkv, 2)
                    scores = torch.einsum("bnhd,bmhd->bhnm", q, k) * module.scale
                    attn = scores.softmax(dim=-1)
                    if num_register_tokens:
                        keep = torch.cat([
                            torch.zeros(1, dtype=torch.long, device=attn.device),
                            torch.arange(
                                1 + num_register_tokens,
                                N,
                                dtype=torch.long,
                                device=attn.device,
                            ),
                        ])
                        attn = attn.index_select(-1, keep).index_select(-2, keep)
                    attentions.append(attn.detach())

                attention_hooks = [
                    block.attn.register_forward_pre_hook(attention_hook)
                    for block in blocks
                ]
            try:
                if "txt" in self.vision_tower_name:
                    device = next(model.parameters()).device
                    _, patch_tokens = model.get_class_and_patch_tokens(images.to(device=device)) # we can not do forward because it would pool the patch tokens
                    out =  {"x_prenorm" : patch_tokens}
                else:
                    out =  model.forward_features(imgs)

            finally:
                for h in hooks:
                    h.remove()
                for h in attention_hooks:
                    h.remove()
            # import pdb; pdb.set_trace()
            hidden_states = [model.head(h) for h in hidden_states] # apply the head to each hidden state. This is important because the head has a projection and without it, the hidden states would be in a different space than the output features that the router dino uses.
            hidden_states = [h[:, 1+self.config.num_register_tokens:, :] for h in hidden_states] # remove register tokens AFTER passing through head?
            return out, tuple(hidden_states), tuple(attentions)
        # import pdb; pdb.set_trace()
        backbone = self.vision_tower

        if type(images) is list:
            if output_attentions:
                raise ValueError("DINO attention collection expects batched tensor images, not a list.")
            image_features = []
            hidden_states_all = []
            for image in images:
                out, hs, _ = _forward_features(backbone, image.to(device=self.device, dtype=self.dtype).unsqueeze(0))
                image_features.append(self.feature_select(out).to(image.dtype))
                if hs is not None:
                    hidden_states_all.append(hs)
        else:
            # import pdb; pdb.set_trace()
            # print("self.device : ", images.to(device=self.device, dtype=self.dtype).device)
            out, hidden_states_all, attentions_all = _forward_features(backbone, images.to(device=self.device, dtype=self.dtype))
            image_features = self.feature_select(out).to(images.dtype)

        if output_attentions:
            return SimpleNamespace(
                hidden_states=hidden_states_all,
                attentions=attentions_all,
            )
        if output_hidden_states:
            return image_features, hidden_states_all
        return image_features


    @property
    def dummy_feature(self):
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        return next(self.vision_tower.parameters()).dtype  # works for any nn.Module

    @property
    def device(self):
        return next(self.vision_tower.parameters()).device  # works for any nn.Module

    @property
    def hidden_size(self):
        return 1024

    @property
    def num_patches(self):
        # 224 / 14 = 16, 16*16 = 256 patch tokens
        # registers are filtered out in feature_select, so this stays 256
        return 256

    @property
    def num_patches_per_side(self):
        return self.config.image_size // self.config.patch_size


class DinoVisionTextTower(DINOVisionTower):  #
    def __init__(self, vision_tower_dino, vision_tower_clip_name, args, delay_load=False):
        self.text_tower = None
        # self.text_tokenizer = None
        self.vision_tower_name_dino = vision_tower_dino
        self.vision_tower_name_clip = vision_tower_clip_name
        super().__init__(vision_tower_dino, args, delay_load)

    def load_model(self, device_map=None):
        super().load_model()
        if self.text_tower is None:
            # import pdb; pdb.set_trace()
            self.text_tower = torch.hub.load(
                'facebookresearch/dinov2',
                'dinov2_vitl14_reg4_dinotxt_tet1280d20h24l'
            ).text_model
            # is it fine to just use ClipTokenizer?
            self.text_tokenizer = CLIPTokenizer.from_pretrained(self.vision_tower_name_clip)
            self.text_tokenizer.pad_token_id = 0  # DINOv2 pads with 0, not EOS (49407)
            self.text_tower.requires_grad_(False)

    def get_text_tower(self):  # NOTE: this will be used to return a handler to self.text_tower
        if self.text_tower is None:
            raise RuntimeError("Text tower not loaded. Please call load_model() first.")
        return self.text_tower

    def encode_text(self, input_ids, attention_mask=None):
        # import pdb; pdb.set_trace()
        # for dino txt, we are not using the ClipTextTower.
        if "txt" in self.vision_tower_name_dino:
            # import pdb; pdb.set_trace()
            # could be easier to just manually do the forward pass for dino txt
            text_tokens = self.text_tower.backbone(input_ids)
            text_tokens = self.text_tower.head(text_tokens)
            # dino does pooling, so we need to do that too! The config of dino uses
            # self.tokens_pooler_type == "first", so I will just use "first" as well
            if self.text_tower.tokens_pooler_type == "first":
                return text_tokens[:, 0, :]
            elif self.text_tower.tokens_pooler_type == "last":
                return text_tokens[:, -1, :]
            elif self.text_tower.tokens_pooler_type == "argmax":
                return text_tokens[torch.arange(text_tokens.shape[0]), input_ids.argmax(dim=-1)] # torch.Size([32, 2048])
            else:
                raise ValueError(f"Unknown pooler type: {self.text_tower.tokens_pooler_type}")
        else:
            text_out = self.get_text_tower()(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            return text_out.last_hidden_state # [32, 27, 768]
