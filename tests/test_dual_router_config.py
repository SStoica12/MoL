"""Keep CLIP and DINO text widths separate when restoring dual routers."""
import unittest
from types import SimpleNamespace

import torch
from llava.model.builder import _dino_router_config, _configure_dual_router_features
from llava.model.router import TextConditionedRouter


class DualRouterConfigTests(unittest.TestCase):
    def test_dual_router_restores_dino_text_width_without_changing_clip(self):
        config = SimpleNamespace(encoder_type='clip-dinov2_txt',
                                 mm_text_hidden_size=768, dino_mm_text_hidden_size=2048,
                                 mm_hidden_size=4, hybrid_reserve_mode='disagreement_gate')
        branch = _dino_router_config(config)
        router = TextConditionedRouter(
            branch, torch.zeros(256, 4), text_conditioning='attention',
            text_attention_type='qk', text_attention_dim=8, attn_level='hybrid',
            save_router_stats=False,
        )
        self.assertEqual(tuple(router.text_attn_q.weight.shape), (8, 2048))
        self.assertEqual(tuple(router.hybrid_beta_gate.weight.shape), (1, 2048))
        self.assertEqual(config.mm_text_hidden_size, 768)
        self.assertEqual(config.encoder_type, 'clip-dinov2_txt')

    def test_dual_clip_drops_cls_while_single_encoder_selection_is_preserved(self):
        from llava.model.router import Router
        for encoder in ('clip-dinov2_txt', 'clip', 'siglip'):
            config = SimpleNamespace(encoder_type=encoder)
            clip = Router(config, torch.zeros(576, 4), save_router_stats=False)
            dino = Router(config, torch.zeros(256, 4), save_router_stats=False)
            model = SimpleNamespace(config=config, router_clip=clip, router_dino=dino)
            _configure_dual_router_features(model)
            tokens = torch.randn(1, 577, 4)
            actual = clip._select_vision_features(tokens)
            expected = tokens[:, 1:] if encoder == 'clip-dinov2_txt' else tokens
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            self.assertEqual(dino.vision_feature_select_strategy, 'cls_patch')

    def test_single_dino_configuration_is_preserved(self):
        config = SimpleNamespace(encoder_type='dinov2_txt',
                                 mm_text_hidden_size=2048, dino_mm_text_hidden_size=2048)
        branch = _dino_router_config(config)
        self.assertEqual(branch.mm_text_hidden_size, config.mm_text_hidden_size)
        self.assertEqual(branch.dino_mm_text_hidden_size, config.dino_mm_text_hidden_size)
        self.assertEqual(config.encoder_type, 'dinov2_txt')


if __name__ == '__main__':
    unittest.main()
