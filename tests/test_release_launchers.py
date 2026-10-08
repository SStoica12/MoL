"""Launch configuration contracts. No weights, network, or GPUs required."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]


def module(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / (name + '.py'))
    loaded = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(loaded)
    return loaded


train = module('train')
evaluate = module('evaluate')


class LauncherTests(unittest.TestCase):
    def training_args(self, profile, stage='finetune'):
        return train.parser().parse_args([
            '--profile', profile, '--stage', stage,
            '--data-path', 'data/train.json', '--image-folder', 'data/images',
            '--output-dir', 'checkpoints/new', '--pretrain-dir', 'checkpoints/pretrain',
        ])

    def test_branch_specific_stage_one_files(self):
        for profile, expected in [
            ('vicuna_clip_hybrid', {'pretrain_mm_mlp_adapter'}),
            ('vicuna_dinov2_hybrid', {'pretrain_dino_mm_mlp_adapter'}),
            ('vicuna_clip_dinov2_hybrid', {'pretrain_mm_mlp_adapter', 'pretrain_dino_mm_mlp_adapter'}),
        ]:
            with self.subTest(profile=profile):
                _, options = train.command(self.training_args(profile))
                actual = {k for k in options if k.startswith('pretrain_')}
                self.assertEqual(actual, expected)
                self.assertEqual(options['fusion_type'], train.PROFILES[profile]['fusion_type'])

    def test_stage_one_freezes_backbone_stage_two_uses_lora_for_llama3(self):
        _, first = train.command(self.training_args('llama3-8b_siglip_patch', 'pretrain'))
        _, second = train.command(self.training_args('llama3-8b_siglip_patch'))
        self.assertTrue(first['freeze_backbone'])
        self.assertTrue(first['tune_mm_mlp_adapter'])
        self.assertFalse(second['tune_mm_mlp_adapter'])
        self.assertTrue(second['lora_enable'])
        self.assertEqual(second['num_layers_to_choose'], 4)
        self.assertEqual(second['version'], 'llama')

    def test_training_options_are_accepted_by_existing_trainer(self):
        from transformers import HfArgumentParser
        from llava.train.train_router import ModelArguments, DataArguments, TrainingArguments
        p = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
        for profile, cfg in train.PROFILES.items():
            if not cfg['runtime_supported']:
                continue
            for stage in ('pretrain', 'finetune'):
                with self.subTest(profile=profile, stage=stage):
                    cmd, _ = train.command(self.training_args(profile, stage))
                    args_start = cmd.index('--model_name_or_path')
                    p.parse_args(cmd[args_start:])

    def eval_args(self, folder, benchmark='vstar'):
        return evaluate.parser().parse_args([
            '--model-path', str(folder), '--benchmark', benchmark,
            '--question-file', 'data/questions.jsonl', '--image-folder', 'data/images',
            '--output-dir', 'outputs/new', '--dry-run',
        ])

    def test_renamed_adapter_uses_config_base_and_keeps_checkpoint_k(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp) / 'arbitrary-directory-name'
            folder.mkdir()
            (folder / 'config.json').write_text(json.dumps({'model_type': 'llava', 'num_layers_to_choose': 4}))
            (folder / 'adapter_config.json').write_text(json.dumps({
                'base_model_name_or_path': 'meta-llama/Meta-Llama-3-8B-Instruct'}))
            cmd, score = evaluate.commands(self.eval_args(folder))
            self.assertEqual(cmd[cmd.index('--conv-mode') + 1], 'llama')
            self.assertEqual(cmd[cmd.index('--model-base') + 1], 'meta-llama/Meta-Llama-3-8B-Instruct')
            self.assertNotIn('--num_layers_to_choose', cmd)
            self.assertNotIn('--judge-model', score)
            self.assertEqual(json.loads((folder / 'config.json').read_text())['num_layers_to_choose'], 4)
            args = self.eval_args(folder)
            args.model_base = '/local/base-model'
            cmd, _ = evaluate.commands(args)
            self.assertEqual(cmd[cmd.index('--conv-mode') + 1], 'llama')

    def test_phi_uses_bunny_template_and_lora_training(self):
        _, options = train.command(self.training_args('phi1_5_siglip_patch'))
        self.assertEqual(options['version'], 'bunny')
        self.assertTrue(options['lora_enable'])
        self.assertIn('pretrain_mm_mlp_adapter', options)
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)
            (folder / 'config.json').write_text(json.dumps({'model_type': 'phi'}))
            (folder / 'adapter_config.json').write_text(json.dumps({
                'base_model_name_or_path': 'microsoft/phi-1_5'}))
            cmd, _ = evaluate.commands(self.eval_args(folder))
            self.assertEqual(cmd[cmd.index('--conv-mode') + 1], 'bunny')
            self.assertEqual(cmd[cmd.index('--model-base') + 1], 'microsoft/phi-1_5')

    def test_scorer_arguments_match_existing_entrypoints(self):
        import ast
        for benchmark, scorer in evaluate.SCORERS.items():
            with self.subTest(benchmark=benchmark), tempfile.TemporaryDirectory() as tmp:
                folder = Path(tmp)
                (folder / 'config.json').write_text('{"model_type": "llava"}')
                commands = evaluate.commands(self.eval_args(folder, benchmark))
                for cmd, source in zip(commands, ['model_vqa_loader', 'eval_' + scorer]):
                    tree = ast.parse((ROOT / 'llava/eval' / (source + '.py')).read_text())
                    flags = {arg.value for n in ast.walk(tree)
                             if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'add_argument'
                             for arg in n.args if isinstance(arg, ast.Constant) and isinstance(arg.value, str)}
                    used = {a for a in cmd[3:] if a.startswith('--')}
                    self.assertFalse(used - flags, used - flags)


if __name__ == '__main__':
    unittest.main()
