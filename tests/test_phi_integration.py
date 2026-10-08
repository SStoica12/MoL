"""Small Phi/SigLIP regression tests; no downloads or GPU required."""
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
from torch import nn
from transformers import PhiConfig, SiglipVisionConfig, SiglipVisionModel
from peft import LoraConfig, get_peft_model

from llava.model.language_model.llava_phi import RouterLlavaPhiForCausalLM
from llava.model.phi_builder import normalize_phi_adapter_state, restore_phi_adapter
from llava.train.train_router import find_all_linear_names


class TinyTower(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_tower = SiglipVisionModel(SiglipVisionConfig(
            hidden_size=8, intermediate_size=16, num_hidden_layers=3,
            num_attention_heads=2, image_size=4, patch_size=2,
        ))
        self.text_tower = nn.Embedding(8, 8)
        self.requires_grad_(False)
        self.hidden_size = 8
        self.num_patches = 4
        self.num_patches_per_side = 2
        self.is_loaded = True

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    @staticmethod
    def text_tokenizer(texts, **kwargs):
        ids = torch.tensor([[1, 2, 3] for _ in texts])
        return {'input_ids': ids, 'attention_mask': torch.ones_like(ids)}

    def encode_text(self, input_ids, attention_mask):
        return self.text_tower(input_ids)


def config(mode):
    return PhiConfig(
        vocab_size=64, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=4,
        max_position_embeddings=128, bos_token_id=1, eos_token_id=2, pad_token_id=0,
        encoder_type='siglip', fusion_type='attn', mm_vision_tower='tiny-siglip',
        mm_hidden_size=8, mm_text_hidden_size=8, mm_projector_type='mlp2x_gelu',
        mm_use_text_enc=True, mm_use_im_start_end=False, mm_vision_select_feature='patch',
        mm_patch_merge_type='flat', use_text_conditioned_router=True,
        text_attention_dim=8, text_conditioning='attention', text_attention_type='qk',
        attn_level=mode, num_layers_to_choose=1 if mode=='patch_layer' else 2,
        force_reserve_layer=True, reserve_layer=-2, learnable_gate=False,
        hybrid_reserve_mode='local_gate' if mode=='patch_layer' else 'disagreement_gate',
        save_router_stats=False,
    )


def make_model(mode):
    with patch('llava.model.llava_arch.build_vision_tower', side_effect=lambda *a, **k: TinyTower()):
        return RouterLlavaPhiForCausalLM(config(mode))


class PhiIntegrationTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.ids = torch.tensor([[7, -200, 8, 9]])
        self.labels = torch.tensor([[-100, -100, 8, 9]])
        self.images = torch.randn(1, 3, 4, 4)

    def test_patch_and_hybrid_forward_backward_and_cached_generation(self):
        for mode in ('patch_layer', 'hybrid'):
            with self.subTest(mode=mode):
                model = make_model(mode)
                self.assertEqual(tuple(model.router_clip.weights.shape), (4, 8))
                out = model(input_ids=self.ids, images=self.images, clip_texts=['What color?'],
                            labels=self.labels, use_cache=False)
                # Preserve the shared router's configured patch selection.
                self.assertEqual(tuple(out.logits.shape), (1, 6, 64))
                loss = out.loss + model.load_balance_alpha * model.get_aux_loss(self.images)
                self.assertTrue(torch.isfinite(loss))
                loss.backward()
                for module in (model.router_clip.text_attn_q, model.model.mm_projector[0]):
                    self.assertIsNotNone(module.weight.grad)
                    self.assertTrue(torch.isfinite(module.weight.grad).all())
                    self.assertGreater(float(module.weight.grad.abs().sum()), 0)
                model.eval()
                ids = model.generate(self.ids, images=self.images, prompts=['What color?'],
                                     max_new_tokens=3, do_sample=False, use_cache=True)
                self.assertGreater(ids.numel(), 0)

    def test_legacy_and_current_multimodal_weights_restore_exactly(self):
        for mode in ('patch_layer', 'hybrid'):
            model = make_model(mode)
            state = {k: v.detach().clone() for k,v in model.state_dict().items()
                     if k.startswith(('model.mm_projector.', 'router_clip.'))}
            legacy = {}
            for key,value in state.items():
                if key in ('router_clip.hybrid_reserve_d_weight', 'router_clip.hybrid_reserve_d_bias'):
                    value = value.reshape(1)
                key = key.replace('router_clip._weight_fn.weights', 'router.weights')
                key = key.replace('router_clip.', 'router.')
                legacy['base_model.model.'+key] = value
            restored = make_model(mode)
            restore_phi_adapter(restored, legacy)
            for key,value in state.items():
                torch.testing.assert_close(restored.state_dict()[key],value,rtol=0,atol=0)
            restore_phi_adapter(restored, {'base_model.model.'+k:v for k,v in state.items()})
            bad = dict(legacy)
            del bad['base_model.model.router.text_attn_q.weight']
            with self.assertRaisesRegex(ValueError, 'missing='):
                restore_phi_adapter(restored, bad)

    def test_phi_lora_targets_and_adapter_gradients(self):
        model = make_model('hybrid')
        targets = find_all_linear_names(model)
        self.assertEqual(set(targets), {'q_proj','k_proj','v_proj','dense','fc1','fc2'})
        wrapped = get_peft_model(model, LoraConfig(
            r=2, lora_alpha=4, target_modules=targets, task_type='CAUSAL_LM',
        ))
        wrapped.get_base_model().router_clip.requires_grad_(True)
        wrapped.get_model().mm_projector.requires_grad_(True)
        out = wrapped(input_ids=self.ids, images=self.images, clip_texts=['What color?'], labels=self.labels)
        loss = out.loss + wrapped.load_balance_alpha * wrapped.get_aux_loss(self.images)
        loss.backward()
        self.assertTrue(any(p.grad is not None and p.grad.abs().sum()>0
                            for n,p in wrapped.named_parameters() if 'lora_B' in n))
        self.assertTrue(any(n.startswith('base_model.model.router_clip.') for n,_ in wrapped.named_parameters()))
        self.assertFalse(any(n.startswith('router_clip.') for n,_ in wrapped.named_parameters()))

    def test_phi_text_only_forward_preserves_native_vocabulary(self):
        model = make_model('patch_layer')
        ids = torch.tensor([[7, 8, 9]])
        result = model(input_ids=ids, labels=ids)
        self.assertEqual(result.logits.shape[-1], 64)
        self.assertTrue(torch.isfinite(result.loss))

    def test_training_wrapper_keeps_phi_auxiliary_loss_and_gradients(self):
        from llava.train.llava_trainer import RouterLLaVATrainer

        class ForwardOnlyWrapper(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, **kwargs):
                return self.module(**kwargs)

        model = make_model('hybrid')
        wrapped = ForwardOnlyWrapper(model)
        trainer = object.__new__(RouterLLaVATrainer)
        trainer.model = model
        trainer.accelerator = SimpleNamespace(unwrap_model=lambda wrapper: wrapper.module)
        trainer.encoder_type = 'siglip'
        trainer.fusion_type = 'attn'
        trainer.aux_losses = []
        loss, output = trainer.compute_loss(wrapped, dict(
            input_ids=self.ids, labels=self.labels, images=self.images,
            clip_texts=['What color?'],
        ), return_outputs=True)
        self.assertTrue(torch.isfinite(loss))
        self.assertGreater(trainer.last_aux_loss, 0)
        loss.backward()
        self.assertGreater(float(model.router_clip.text_attn_q.weight.grad.abs().sum()), 0)

    def test_full_checkpoint_roundtrip_reconstructs_router(self):
        for mode in ('patch_layer', 'hybrid'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as tmp:
                model = make_model(mode).eval()
                model.save_pretrained(tmp)
                with patch('llava.model.llava_arch.build_vision_tower', side_effect=lambda *a, **k: TinyTower()):
                    restored = RouterLlavaPhiForCausalLM.from_pretrained(tmp).eval()
                with torch.no_grad():
                    args = dict(input_ids=self.ids, images=self.images, clip_texts=['What color?'])
                    torch.testing.assert_close(model(**args).logits, restored(**args).logits, rtol=0, atol=0)
                self.assertEqual(model.router_clip.num_layers_to_choose, restored.router_clip.num_layers_to_choose)

    def test_phi_bunny_labels_only_supervise_assistant_replies(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, decoders
        from transformers import PreTrainedTokenizerFast
        from llava.train.phi_utils import preprocess_phi
        from llava.constants import IGNORE_INDEX
        vocab = {c: i for i,c in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))}
        vocab['<|endoftext|>'] = len(vocab)
        backend = Tokenizer(models.BPE(vocab=vocab, merges=[]))
        backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        backend.decoder = decoders.ByteLevel()
        tok = PreTrainedTokenizerFast(
            tokenizer_object=backend, bos_token='<|endoftext|>', eos_token='<|endoftext|>',
            pad_token='<|endoftext|>', unk_token='<|endoftext|>', model_max_length=2048,
        )
        source = [[{'from':'human','value':'<image>\nWhat color?'},
                   {'from':'gpt','value':'Red.'},
                   {'from':'human','value':'Is it blue?'},
                   {'from':'gpt','value':'No.'}]]
        result = preprocess_phi(source, tok)
        target = result['labels'][0]
        self.assertEqual(tok.decode(target[target != IGNORE_INDEX]), ' Red.<|endoftext|> No.<|endoftext|>')
        self.assertEqual(int(target[result['input_ids'][0] == -200][0]), IGNORE_INDEX)


if __name__ == '__main__':
    unittest.main()
