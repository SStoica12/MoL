# Mixture of Layers (MoL)

**Mixture of Layers: Dynamic Layer Routing for Visual Reasoning**

[Paper](https://arxiv.org/abs/2610.09440) · [Project page](https://wjdghks950.github.io/mol.github.io/) · [Model weights](https://huggingface.co/wjdghks950)

MoL selects intermediate vision-encoder layers according to the input instruction. Its sparse router can select layers for an entire image (`MoL_layer`), for individual patches (`MoL_patch`), or combine global and local routing (`MoL_hybrid`). The selected visual features are projected into the language model, helping it access the details needed for a particular question. The dual-encoder variant combines routed CLIP and DINOv2 features through upsampling, concatenation, and a learned projection.

This repository contains the MoL model, training entry point, dataset converters, and benchmark evaluators. The Python package keeps the `llava` name inherited from LLaVA.

## Install

Use Linux with NVIDIA GPUs. Training uses BF16, DeepSpeed, and FlashAttention 2; it needs a compatible CUDA toolkit (`nvcc`), compiler, and PyTorch installation. The examples use a single node with eight visible GPUs. Adjust microbatch size and gradient accumulation for your hardware; no fixed minimum GPU memory is guaranteed.

From the repository root:

```bash
conda env create -f environment.yml
conda activate mol
python -m pip install -e '.[train,eval]'
python -m pip install flash-attn==2.5.2 --no-build-isolation
```

For evaluation only, install `'.[eval]'` instead; FlashAttention is required by the training entry point. The principal pins are Python 3.10, PyTorch 2.1.2, Transformers 4.37.2, NumPy 1.26.4, PEFT 0.10.0, Accelerate 0.21.0, and DeepSpeed 0.12.6. The complete dependencies are in [pyproject.toml](pyproject.toml). These versions follow the historical training environments; upgrading Transformers or PyTorch independently can break the custom model classes.

CPU regression checks are available below. Both released Phi checkpoints have also passed GPU loading, image generation, and backward checks. See [compatibility notes](docs/COMPATIBILITY.md) for validation scope and remaining limitations.

## Download checkpoints

All eight repositories below are public. `k` is stored as `num_layers_to_choose` in each checkpoint's `config.json`, so checkpoint names do not include a `k` suffix.

| Backbone / vision encoder | MoL variant | k | Download | Format |
| --- | --- | --- | --- | --- |
| Vicuna-13B / CLIP | patch | 1 | [vicuna_clip_patch](https://huggingface.co/wjdghks950/vicuna_clip_patch) | Full model |
| Vicuna-13B / CLIP | hybrid | 1 | [vicuna_clip_hybrid](https://huggingface.co/wjdghks950/vicuna_clip_hybrid) | Full model |
| Vicuna-13B / DINOv2 + text | patch | 2 | [vicuna_dinov2_patch](https://huggingface.co/wjdghks950/vicuna_dinov2_patch) | Full model |
| Vicuna-13B / DINOv2 + text | hybrid | 2 | [vicuna_dinov2_hybrid](https://huggingface.co/wjdghks950/vicuna_dinov2_hybrid) | Full model |
| Llama-3-8B-Instruct / SigLIP | patch | 4 | [llama3-8b_siglip_patch](https://huggingface.co/wjdghks950/llama3-8b_siglip_patch) | LoRA + multimodal weights |
| Llama-3-8B-Instruct / SigLIP | hybrid | 1 | [llama3-8b_siglip_hybrid](https://huggingface.co/wjdghks950/llama3-8b_siglip_hybrid) | LoRA + multimodal weights |
| Phi-1.5 / SigLIP | patch | 1 | [phi1_5_siglip_patch](https://huggingface.co/wjdghks950/phi1_5_siglip_patch) | LoRA + multimodal weights |
| Phi-1.5 / SigLIP | hybrid | 2 | [phi1_5_siglip_hybrid](https://huggingface.co/wjdghks950/phi1_5_siglip_hybrid) | LoRA + multimodal weights |

All eight checkpoints are supported by the training/evaluation framework. Phi uses its native language-model architecture with the shared MoL router. No public dual-encoder or global-layer checkpoint is included in this release.

Download a complete model directory:

```bash
hf download wjdghks950/vicuna_clip_hybrid \
  --local-dir checkpoints/vicuna_clip_hybrid
```

LoRA checkpoints additionally require the base model named in `adapter_config.json`: `meta-llama/Meta-Llama-3-8B-Instruct` for Llama-3 and `microsoft/phi-1_5` for Phi. Obtain access to the Meta model and run `hf auth login` before using it. Keep `adapter_model.safetensors`, `non_lora_trainables.bin`, and configuration files together; the tokenizer comes from the base model. The MoL evaluator reads the adapter base automatically; `--model-base` can point to a local copy. These checkpoints require this repository's custom loader, not a standalone generic `AutoModelForCausalLM` call. Vision/text encoder weights are fetched separately on first use.

## Prepare data

Follow [docs/DATA.md](docs/DATA.md) for download links, extraction layout, and benchmark conversion commands.

- Vicuna uses LLaVA's 558K caption alignment data and 665K instruction mixture.
- The Bunny-based Llama-3/SigLIP and Phi-1.5/SigLIP setups use Bunny pretraining data and Bunny-695K instruction data.
- Training reads a JSON list of conversations; evaluation reads converted JSONL questions. Image paths must resolve relative to the `--image-folder` passed to the launcher.

The data, checkpoints, generated predictions, and logs are excluded from source control.

## Train

[configs/models.json](configs/models.json) defines the router settings for the released patch/hybrid models, plus global-layer and dual-encoder examples. Launchers only assemble arguments for the existing trainer; they do not alter routing, fusion, or losses. These are portable training recipes, not a claim of exact historical score reproduction.

The two-stage example below trains Vicuna + CLIP hybrid. Stage 1 trains the projector and router with the language backbone frozen; stage 2 instruction-tunes the model. Run both stages with the same profile and `k`.

```bash
python scripts/train.py \
  --profile vicuna_clip_hybrid --stage pretrain \
  --data-path data/pretrain/blip_laion_cc_sbu_558k.json \
  --image-folder data/pretrain/images \
  --output-dir checkpoints/vicuna_clip_hybrid_pretrain \
  --gpus 8 --batch-size 16 --gradient-accumulation 2

python scripts/train.py \
  --profile vicuna_clip_hybrid --stage finetune \
  --pretrain-dir checkpoints/vicuna_clip_hybrid_pretrain \
  --data-path data/train/llava_v1_5_mix665k.json \
  --image-folder data/train \
  --output-dir checkpoints/vicuna_clip_hybrid \
  --gpus 8 --batch-size 8 --gradient-accumulation 2
```

The global batch size is `GPUs × batch-size × gradient-accumulation`. Append `--dry-run` to inspect a command without loading data/models. Use a new output directory for each run. The underlying [trainer](llava/train/train_router.py) handles checkpoint resume; the convenience launcher deliberately requires an empty output directory.

Change `--profile` in **both** commands to select a variant:

| Profiles | Routing / training |
| --- | --- |
| `vicuna_clip_patch`, `vicuna_clip_hybrid` | Single CLIP encoder; full instruction finetuning |
| `vicuna_dinov2_patch`, `vicuna_dinov2_hybrid` | DINOv2 with aligned text encoder; full instruction finetuning |
| `vicuna_clip_layer`, `vicuna_dinov2_layer` | Example global-layer router configurations |
| `llama3-8b_siglip_patch`, `llama3-8b_siglip_hybrid` | SigLIP; LoRA instruction finetuning, rank 128 / alpha 256; use Bunny data paths |
| `phi1_5_siglip_patch`, `phi1_5_siglip_hybrid` | SigLIP; LoRA instruction finetuning, rank 128 / alpha 256; use Bunny data paths and the automatically selected `bunny` template |
| `vicuna_clip_dinov2_hybrid` | CLIP + DINOv2, `fusion_type=concate_upsample` |

For DINOv2, stage 1 saves `dino_mm_projector.bin`. CLIP/SigLIP saves `mm_projector.bin`. Dual-encoder training saves and requires **both** files; its CLIP adapter also contains the learned fusion projection. The launcher selects the appropriate files from `--pretrain-dir`.

These files must contain **MoL router weights**. A stock LLaVA/Bunny projector does not satisfy the current stage-2 loader, and a Hugging Face repository ID is not a local `.bin` path. Start with MoL stage 1 as shown above. Changing architecture or routing settings between stages can cause incompatible state dictionaries.

Options include `--k`, `--learning-rate`, `--router-lr`, `--epochs`, `--model-base`, `--master-port`, and `--workers`. Stage 1 defaults to projector LR `1e-3`; the router LR and stage-2 LR come from the profile. Vision backbones remain frozen in these recipes.

## Evaluate

Prepare a benchmark using [the conversion commands](docs/DATA.md#evaluation-data), then run:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-path checkpoints/vicuna_clip_hybrid \
  --benchmark vstar \
  --question-file data/eval/vstar/questions.jsonl \
  --image-folder data/eval/vstar \
  --output-dir outputs/vicuna_clip_hybrid/vstar
```

Choose `vstar`, `mmstar`, `hrbench4k`, `hrbench8k`, `realworldqa`, `naturalbench`, or `charxiv`. Each run invokes the existing generator and scorer, writing `answers.jsonl`, `metrics.json`, and `details.jsonl` under the specified output directory. The loader restores router settings and `k` from the checkpoint config. Use `--k` only for an intentional evaluation-time override. Add `--dry-run` to inspect both commands; the downloaded checkpoint config must exist.

For a Llama-3 or Phi adapter, download its complete directory and use the same command with its `--model-path`. The wrapper reads the base model from `adapter_config.json` and selects `llama` for Llama-3, `bunny` for Phi, or `vicuna_v1` for Vicuna. For example, use `hf download wjdghks950/phi1_5_siglip_hybrid --local-dir checkpoints/phi1_5_siglip_hybrid`, then evaluate with `--model-path checkpoints/phi1_5_siglip_hybrid`.

CharXiv descriptive questions can be scored locally. Reasoning questions need judge grades; without them the summary is incomplete. Explicitly opt into API grading with `--judge-model YOUR_JUDGE_MODEL` and set `OPENAI_API_KEY`. This sends predictions/questions to the selected API and incurs its usage costs. Use a consistent judge and dataset split when comparing models. See [compatibility notes](docs/COMPATIBILITY.md) for evaluation protocol details.

## Code map and checks

- [llava/model/router.py](llava/model/router.py): image, patch, and hybrid routing.
- [llava/model/llava_arch.py](llava/model/llava_arch.py): multimodal inputs and dual-encoder fusion.
- [llava/model/multimodal_encoder](llava/model/multimodal_encoder): CLIP, DINOv2, and SigLIP towers.
- [llava/train/train_router.py](llava/train/train_router.py): training and adapter loading/saving.
- [llava/eval](llava/eval): generation and benchmark scoring.
- [scripts](scripts): launchers, data converters, and DeepSpeed configurations.

```bash
python -m unittest discover -s tests -v
```

The CPU tests check dual-encoder fusion, single-encoder behavior, Phi forward/backward and cached generation, LoRA gradients, assistant-only training labels, checkpoint round trips, and launcher contracts. Full training and benchmark evaluation require models, datasets, and GPUs.

## Citation and acknowledgments

```bibtex
@article{kim2026mixture,
  title={Mixture of Layers: Dynamic Layer Routing for Visual Reasoning},
  author={Kim, Jeonghwan and Stoica, Sofia and Chung, Jiwan and Blume, Ansel and Ha, Hyeonjeong and Wang, Zhenhailong and Dong, Xin Luna and Ji, Heng},
  year={2026},
  eprint={2610.09440},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  journal={arXiv preprint arXiv:2610.09440},
  url={https://arxiv.org/abs/2610.09440},
}
```

Built on [LLaVA](https://github.com/haotian-liu/LLaVA), [Bunny](https://github.com/BAAI-DCAI/Bunny), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), and [SigLIP](https://github.com/google-research/big_vision). Source code uses the [Apache-2.0 license](LICENSE); datasets and pretrained weights retain their respective licenses.
