"""CPU regression tests; isolate model code from optional vision/LLM dependencies.

Run: python -m unittest discover -s tests -p 'test_dual_encoder_concat.py'
"""
import ast
from abc import ABC, abstractmethod
from pathlib import Path
from types import SimpleNamespace
import re
import unittest
import tempfile

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]


def definitions(path, names, namespace):
    tree = ast.parse((ROOT / path).read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef)) and n.name in names]
    exec(compile(tree, path, 'exec'), namespace)
    return namespace


projectors = definitions('llava/model/multimodal_projector/builder.py',
                         {'build_vision_projector', 'IdentityMap', 'SimpleResBlock'},
                         dict(torch=torch, nn=nn, re=re))
arch = definitions('llava/model/llava_arch.py',
                   {'uses_concat_upsample', 'concat_upsample_features', 'LlavaMetaModel', 'LlavaMetaForCausalLM'},
                   dict(torch=torch, nn=nn, F=F, ABC=ABC, abstractmethod=abstractmethod,
                        IGNORE_INDEX=-100, IMAGE_TOKEN_INDEX=-200, rank0_print=lambda *args: None,
                        build_vision_projector=projectors['build_vision_projector'],
                        build_vision_tower=lambda *args, **kwargs: nn.Identity()))


def config(encoder='clip-dinov2_txt', fusion='concate_upsample'):
    return SimpleNamespace(encoder_type=encoder, fusion_type=fusion, hidden_size=8,
                           mm_hidden_size=6, mm_hidden_size_dino=5, mm_projector_type='mlp2x_gelu')


class Base(nn.Module):
    def __init__(self, config):
        super().__init__()


class Meta(arch['LlavaMetaModel'], Base):
    pass


class Harness(arch['LlavaMetaForCausalLM']):
    def __init__(self, cfg):
        self.config = cfg
        self.device = torch.device('cpu')
        self.model = Meta(cfg)
        self.model.embed_tokens = nn.Embedding(16, 8)

    def get_model(self):
        return self.model

    def get_vision_tower(self):
        return True

    def get_dino_vision_tower(self):
        return True

    def encode_images_withclip(self, images, **kwargs):
        return images

    def encode_images_withdino(self, images, **kwargs):
        return images


class ConcatTests(unittest.TestCase):
    def test_projector_dimensions_and_single_encoder_regression(self):
        for encoder in ('clip', 'dinov2_txt', 'siglip', 'clip-dinov2_txt', 'siglip-dinov2_txt'):
            for fusion in ('attn', 'film', 'concate_upsample'):
                for branch, inp in [('clip', 6), ('dino', 5)]:
                    cfg = config(encoder, fusion)
                    torch.manual_seed(7)
                    actual = projectors['build_vision_projector'](cfg, type=branch)
                    dual = 'dino' in encoder and ('clip' in encoder or 'siglip' in encoder)
                    # Preserve legacy single-encoder behavior, even for a stale fusion setting.
                    out = 16 if fusion == 'concate_upsample' and not dual else 8
                    torch.manual_seed(7)
                    expected = nn.Sequential(nn.Linear(inp, 8), nn.GELU(), nn.Linear(8, out))
                    x = torch.randn(2, 4, inp)
                    torch.testing.assert_close(actual(x), expected(x), rtol=0, atol=0)
                    self.assertEqual(actual(x).shape[-1], out)

    def test_historical_fusion_and_gradients(self):
        torch.manual_seed(3)
        clip = torch.randn(2, 576, 8, requires_grad=True)
        dino = torch.randn(2, 256, 8, requires_grad=True)
        projector = nn.Linear(16, 8)
        actual = arch['concat_upsample_features'](clip, dino, projector)
        # Frozen reference operation order from Sofia's llava_arch.py.
        resized = F.interpolate(dino.reshape(2, 8, 16, 16), size=(24, 24),
                                mode='bilinear', align_corners=False).reshape(clip.shape)
        expected = projector(torch.cat([clip, resized], dim=-1))
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actual.square().mean().backward()
        for tensor in (clip, dino, projector.weight, projector.bias):
            self.assertIsNotNone(tensor.grad)
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(tensor.grad.abs().sum().item(), 0)

    def test_checkpoint_keys_and_roundtrip(self):
        cfg = config()
        model = Meta(cfg)
        self.assertEqual(tuple(model.state_dict()['downsample_image_features.weight'].shape), (8, 16))
        restored = Meta(cfg)
        restored.load_state_dict(model.state_dict(), strict=True)
        for encoder in ('clip', 'dinov2_txt', 'siglip'):
            self.assertFalse(hasattr(Meta(config(encoder)), 'downsample_image_features'))
        for fusion in ('attn', 'film'):
            self.assertFalse(hasattr(Meta(config(fusion=fusion)), 'downsample_image_features'))

    def test_pretrained_adapter_restores_fusion_without_reinitializing(self):
        class Tower(nn.Module):
            hidden_size = 6

            def load_model(self):
                pass

        cfg = config()
        model = Meta(cfg)
        model.vision_tower = Tower()
        model.dino_tower = Tower()
        model.mm_projector = projectors['build_vision_projector'](cfg)
        model.dino_mm_projector = projectors['build_vision_projector'](cfg, type='dino')
        original = model.downsample_image_features
        with tempfile.TemporaryDirectory() as directory:
            clip_path, dino_path = Path(directory)/'mm_projector.bin', Path(directory)/'dino_mm_projector.bin'
            clip_weights = {'model.mm_projector.'+k: v for k, v in model.mm_projector.state_dict().items()}
            clip_weights.update({'model.downsample_image_features.weight': torch.full((8,16), 0.25),
                                 'model.downsample_image_features.bias': torch.full((8,), 0.5)})
            torch.save(clip_weights, clip_path)
            torch.save({'model.dino_mm_projector.'+k: v for k, v in model.dino_mm_projector.state_dict().items()}, dino_path)
            args = SimpleNamespace(vision_tower='stub', mm_vision_select_layer=-2,
                                   mm_vision_select_feature='patch', encoder_type=cfg.encoder_type,
                                   pretrain_mm_mlp_adapter=str(clip_path), pretrain_dino_mm_mlp_adapter=str(dino_path),
                                   mm_patch_merge_type='flat', mm_use_text_enc=False, mm_projector_type='mlp2x_gelu')
            model.initialize_vision_modules(args)
            self.assertIs(model.downsample_image_features, original)
            torch.testing.assert_close(original.weight, torch.full((8,16), 0.25))
            torch.testing.assert_close(original.bias, torch.full((8,), 0.5))
            # An old adapter lacking fusion weights must fail instead of silently randomizing them.
            del clip_weights['model.downsample_image_features.weight']
            del clip_weights['model.downsample_image_features.bias']
            torch.save(clip_weights, clip_path)
            with self.assertRaises(RuntimeError):
                model.initialize_vision_modules(args)

    def test_multimodal_insertion_and_single_encoder_outputs(self):
        for encoder in ('clip-dinov2_txt', 'clip', 'dinov2_txt', 'siglip'):
            cfg = config(encoder, 'concate_upsample' if '-' in encoder else 'attn')
            h = Harness(cfg)
            clip = torch.randn(2, 9, 8) if 'clip' in encoder or 'siglip' in encoder else None
            dino = torch.randn(2, 4, 8) if 'dino' in encoder else None
            ids = torch.tensor([[1, -200, 2], [3, -200, 4]])
            out = h.prepare_inputs_labels_for_multimodal_withdino(
                ids, labels=ids.clone(), images=clip, dino_images=dino, encoder_type=encoder)
            features = (arch['concat_upsample_features'](clip, dino, h.model.downsample_image_features)
                        if clip is not None and dino is not None else clip if clip is not None else dino)
            expected = torch.cat([h.model.embed_tokens(ids[:, :1]), features,
                                  h.model.embed_tokens(ids[:, -1:])], dim=1)
            torch.testing.assert_close(out[4], expected, rtol=0, atol=0)
            self.assertTrue((out[5][:, 1:-1] == -100).all())


if __name__ == '__main__':
    unittest.main()
