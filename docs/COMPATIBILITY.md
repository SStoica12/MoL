# Compatibility and reproduction notes

This release packages the existing MoL implementation. Cleanup removes unused code, development artifacts, and debugger remnants; it does not revise routing equations, encoder feature selection, fusion, losses, or checkpoint formats.

## Backbone support

The active trainer and evaluation loader instantiate the Llama-based classes for Vicuna and Llama-3. They do not select the retained `RouterLlavaPhiForCausalLM` implementation for Phi. Phi also has older router integration, so simply renaming a model folder does not make it compatible. The two Phi weight repositories are published, but the release launchers reject Phi until its model dispatch/integration is implemented. This is an existing limitation, not a missing pip package.

CLIP, DINOv2 with an aligned text encoder, and SigLIP implementations are retained. DINOv2 uses the `facebookresearch/dinov2` Torch Hub entry `dinov2_vitl14_reg4_dinotxt_tet1280d20h24l`; this downloads upstream code and weights on first use. Record the cached upstream revision for reproducible runs. The `vision_tower` CLIP identifier in DINO-only profiles is inherited configuration metadata; `encoder_type=dinov2_txt` selects the actual DINO tower.

## Dual-encoder fusion

Select `vicuna_clip_dinov2_hybrid`, which sets `encoder_type=clip-dinov2_txt` and the historical spelling `fusion_type=concate_upsample`. The implementation routes each encoder separately, projects both streams to the language hidden width, aligns their token grids, concatenates features, then uses `model.downsample_image_features` to map `2d` to `d`. The interpolation/reshape ordering intentionally matches the existing training implementation and checkpoints. This path does not enable cross-attention or FiLM fusion.

Stage-1 outputs include the CLIP projector/router/fusion tensors in `mm_projector.bin`, and the DINO projector/router tensors in `dino_mm_projector.bin`. Stage 2 requires both files. Missing learned fusion weights are an incompatibility, not a reason to initialize a random projection silently. No dual-encoder weights are included among the eight public checkpoint repositories.

## Checkpoint and recipe settings

Keep the released `config.json` with its weights. Besides `num_layers_to_choose`, routing depends on `attn_level`, text conditioning, reserved-layer settings, gates, and `vision_tower_delta`. In particular, the released CLIP patch model uses delta features, and the released Llama-3 patch model uses `k=4`. Changing these settings is an ablation, not loading the same model.

The profiles copy the released router settings. Global-layer and dual-encoder profiles are runnable recipe configurations without corresponding public weights; they are not verified reconstructions of every historical training argument. The convenience training script uses explicit defaults for optimization, effective batch size, and stage transitions. It trains MoL adapters in stage 1 and restores those adapters in stage 2. Old run directories and mutable experiment scripts are not used as public instructions.

A standard Bunny projector does not contain MoL router parameters. The current trainer strictly restores the router from the supplied adapter file. Therefore the documented recipe does not claim that a standard Bunny projector, or a Hub repository string passed to `torch.load`, can be used as a drop-in stage-1 checkpoint. Exact historical initialization and dataset variants can differ from these portable recipes.

## Evaluation protocols

The evaluator uses greedy decoding by default. The wrapper restores routing settings from the checkpoint and chooses `vicuna_v1` for Vicuna or `llama` for Llama-3 adapters. Keep the base tokenizer, conversation template, dataset split, and `k` consistent across comparisons. Generation currently runs through the existing loader with its device mapping; the wrapper does not implement distributed benchmark sharding.

The included scorers preserve the repository's existing metrics and answer parsing. They are not a replacement implementation of every upstream benchmark protocol. NaturalBench reports grouped metrics as well as sample-level summaries. Historical V* runs may include extra local category folders absent from the public dataset; a fresh public download does not reconstruct those categories.

CharXiv reasoning requires judge grades. Without `--judge-model`, the scorer writes an incomplete summary for ungraded reasoning items. With it, the scorer uses an external OpenAI-compatible judge and caches grades in `grades.json`. Inspect `grade_count`, missing predictions, and judge errors before interpreting an aggregate. To resume grading existing predictions, invoke `python -m llava.eval.eval_charxiv` directly with the same `--grade-file`; the convenience wrapper protects nonempty output directories from overwrite.

## Validation scope

Dependencies are pinned from historical environments. Installation and runtime imports were checked in an isolated CPU environment using Python 3.11 and the pinned PyTorch/Transformers versions; the conda recipe targets the historical Python 3.10 environment. CPU tests cover feature dimensions, numerical agreement of dual fusion with the historical implementation, gradients, checkpoint round trips, and single-encoder behavior, plus launcher contracts. These checks do not establish end-to-end GPU training, FlashAttention installation, full checkpoint loading, or paper-score reproduction on a fresh machine.
