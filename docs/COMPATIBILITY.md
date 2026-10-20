# Compatibility and reproduction notes

This release packages the existing MoL implementation and integrates the native Phi backbone into training and evaluation. The integration preserves routing equations, encoder feature selection, fusion, and loss definitions; existing released adapters load without rewriting their files.

## Backbone support

The trainer and evaluation loaders select Llama-based classes for Vicuna/Llama-3 and `RouterLlavaPhiForCausalLM` for Phi. Phi support covers the single-SigLIP patch and hybrid variants. It uses the same router/projector interfaces and auxiliary loss as the other single-encoder models, with Phi's native decoder, biased output head, and padded 51,200-token vocabulary.

The Phi loader reads the base model from `adapter_config.json`, restores all projector/router tensors, and merges the LoRA adapter. It accepts legacy `router.*` keys and current `router_clip.*` exports, normalizes the legacy hybrid scalar shapes, and constructs the router for SigLIP's 729 patch positions. Missing multimodal tensors raise an error. Special-token IDs come from the base tokenizer; no manual checkpoint edits are needed.

Phi instruction tuning uses the Bunny conversation template and masks user/system text from the language-model loss. The vision/text backbone stays frozen; the router is registered on the Phi model beneath PEFT so forward passes, optimization, and saving use the same parameters. Stage 1 saves the projector and router together in `mm_projector.bin`; stage 2 saves LoRA weights plus `non_lora_trainables.bin`.

CLIP, DINOv2 with an aligned text encoder, and SigLIP implementations are retained. DINOv2 uses the `facebookresearch/dinov2` Torch Hub entry `dinov2_vitl14_reg4_dinotxt_tet1280d20h24l`; this downloads upstream code and weights on first use. Record the cached upstream revision for reproducible runs. The `vision_tower` CLIP identifier in DINO-only profiles is inherited configuration metadata; `encoder_type=dinov2_txt` selects the actual DINO tower.

## Dual-encoder fusion

Select `vicuna_clip_dinov2_hybrid`, which sets `encoder_type=clip-dinov2_txt` and the historical spelling `fusion_type=concate_upsample`. The implementation routes each encoder separately, projects both streams to the language hidden width, aligns their token grids, concatenates features, then uses `model.downsample_image_features` to map `2d` to `d`. The interpolation/reshape ordering intentionally matches the existing training implementation and checkpoints. This path does not enable cross-attention or FiLM fusion.

The dual-encoder loader constructs the DINO router with its own 2,048-wide text features and removes CLIP's CLS token after routing, matching Sofia's evaluated implementation. Single-encoder loading retains its existing feature selection.

Stage-1 outputs include the CLIP projector/router/fusion tensors in `mm_projector.bin`, and the DINO projector/router tensors in `dino_mm_projector.bin`. Stage 2 requires both files. Missing learned fusion weights are an incompatibility, not a reason to initialize a random projection silently. The public `wjdghks950/vicuna_clip_dinov2_hybrid` checkpoint includes both routers, both projectors, and the learned fusion projection.

## Checkpoint and recipe settings

Keep the released `config.json` with its weights. Besides `num_layers_to_choose`, routing depends on `attn_level`, text conditioning, reserved-layer settings, gates, and `vision_tower_delta`. In particular, the released CLIP patch model uses delta features, and the released Llama-3 patch model uses `k=4`. Changing these settings is an ablation, not loading the same model.

The profiles copy the released router settings. Global-layer profiles are recipe configurations without public weights. Training profiles are not verified reconstructions of every historical training argument. The convenience training script uses explicit defaults for optimization, effective batch size, and stage transitions. It trains MoL adapters in stage 1 and restores those adapters in stage 2. Old run directories and mutable experiment scripts are not used as public instructions.

A standard Bunny projector does not contain MoL router parameters. The current trainer strictly restores the router from the supplied adapter file. Therefore the documented recipe does not claim that a standard Bunny projector, or a Hub repository string passed to `torch.load`, can be used as a drop-in stage-1 checkpoint. Exact historical initialization and dataset variants can differ from these portable recipes.

## Evaluation protocols

The evaluator uses greedy decoding by default. The wrapper restores routing settings from the checkpoint and chooses `vicuna_v1` for Vicuna, `llama` for Llama-3, or `bunny` for Phi. Keep the base tokenizer, conversation template, dataset split, and `k` consistent across comparisons. Generation currently runs through the existing loader with its device mapping; the wrapper does not implement distributed benchmark sharding.

The included scorers preserve the repository's existing metrics and answer parsing. They are not a replacement implementation of every upstream benchmark protocol. NaturalBench reports grouped metrics as well as sample-level summaries. Historical V* runs may include extra local category folders absent from the public dataset; a fresh public download does not reconstruct those categories.

CharXiv reasoning requires judge grades. Without `--judge-model`, the scorer writes an incomplete summary for ungraded reasoning items. With it, the scorer uses an external OpenAI-compatible judge and caches grades in `grades.json`. Inspect `grade_count`, missing predictions, and judge errors before interpreting an aggregate. To resume grading existing predictions, invoke `python -m llava.eval.eval_charxiv` directly with the same `--grade-file`; the convenience wrapper protects nonempty output directories from overwrite.

## Validation scope

Dependencies are pinned from historical environments. Validation used an isolated Python 3.11 environment with PyTorch 2.1.2/CUDA 12.1, Transformers 4.37.2, DeepSpeed 0.12.6, and FlashAttention 2.5.2; the conda recipe targets Python 3.10. The 21 CPU tests cover feature dimensions, agreement of dual fusion with the historical implementation, single-encoder behavior, Phi forward/backward and cached generation, LoRA gradients, assistant-only labels, wrapped auxiliary losses, checkpoint round trips, and launcher contracts.

On one H100, both released Phi adapters loaded with every projector/router tensor checked against the saved weights, generated image-conditioned answers, and produced finite BF16 router/projector gradients. A synthetic example also completed Phi hybrid stage-1 pretraining and two steps of LoRA instruction tuning through the public launcher, with confirmed LoRA, router, and projector weight updates. The saved adapter then reloaded through `scripts/evaluate.py`, completing generation and scoring. These are smoke tests; full-dataset training, multi-GPU execution, and reproduction of the paper's benchmark scores were not run.

The dual-encoder package was also loaded and evaluated on the historical 238-question local V* collection, including OCR and GPT4V-hard. Its model card records the fresh and archived scores separately. The 191 direct-attribute/relative-position questions can be reported separately for comparison with the standard V* subset.
