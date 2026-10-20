"""Legacy checkpoint restoration must not depend on the download folder name."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch
from safetensors.torch import save_file
from llava.model.builder import (_router_checkpoint_state, _router_token_count,
                                 _restore_router_state, _dino_router_config)


class RouterCheckpointTests(unittest.TestCase):
    def test_legacy_keys_across_shards_and_saved_token_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shards = {
                'one.safetensors': {'router.weights': torch.ones(577, 4)},
                'two.safetensors': {'router.hybrid_reserve_d_bias': torch.ones(1)},
            }
            index = {}
            for filename, tensors in shards.items():
                save_file(tensors, root / filename)
                index.update({key: filename for key in tensors})
            (root / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': index}))
            state = _router_checkpoint_state(directory, None, 'clip')
            self.assertEqual(_router_token_count(state, 'router_clip', 576), 577)
            self.assertEqual(state['router_clip']['hybrid_reserve_d_bias'].shape, ())
            self.assertEqual(len(state['router_clip']), 2)

    def test_dino_width_uses_saved_encoder_specific_dimension(self):
        config = SimpleNamespace(encoder_type='dinov2_txt', mm_text_hidden_size=5120,
                                 dino_mm_text_hidden_size=2048)
        self.assertEqual(_dino_router_config(config).mm_text_hidden_size, 2048)
        self.assertEqual(config.mm_text_hidden_size, 5120)

    def test_missing_learned_weights_are_rejected(self):
        model = SimpleNamespace(router_clip=torch.nn.Linear(2, 3), router_dino=None)
        with self.assertRaises(ValueError):
            _restore_router_state(model, {})
        with self.assertRaises(RuntimeError):
            _restore_router_state(model, {'router_clip': {'weight': torch.ones(3, 2)}})
        state = {'router_clip': {'weight': torch.ones(3, 2), 'bias': torch.ones(3)}}
        _restore_router_state(model, state)
        torch.testing.assert_close(model.router_clip.weight, state['router_clip']['weight'])

    def test_adapter_prefixes_and_single_dino_legacy_branch(self):
        state = _router_checkpoint_state('', {'base_model.model.router.weights': torch.ones(257, 4)}, 'dinov2_txt')
        self.assertEqual(_router_token_count(state, 'router_dino', 256), 257)


if __name__ == '__main__':
    unittest.main()
