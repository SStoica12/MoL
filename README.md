# Mixture of Layers (MoL)

**Mixture of Layers: Dynamic Layer Routing for Visual Reasoning**

[Paper](https://arxiv.org/abs/2610.09440) · [Project page](https://wjdghks950.github.io/mol.github.io/) · [Model weights](https://huggingface.co/wjdghks950)

MoL routes intermediate vision-encoder layers based on the input instruction, selecting features per image (`MoL_layer`), per patch (`MoL_patch`), or both (`MoL_hybrid`). It supports CLIP, DINOv2, SigLIP, and a dual-encoder CLIP + DINOv2 setup.

## Install

Use Linux with NVIDIA GPUs and a compatible CUDA toolkit/compiler. Run from the repository root:

```bash
conda env create -f environment.yml
conda activate mol
python -m pip install -e '.[train,eval]'
python -m pip install flash-attn==2.5.2 --no-build-isolation
```

For evaluation only, install `'.[eval]'` and skip FlashAttention. Dependencies are pinned in [environment.yml](environment.yml) and [pyproject.toml](pyproject.toml).

## Checkpoints

| Backbone / vision encoder | Patch | Hybrid |
| --- | --- | --- |
| Vicuna-13B / CLIP | [Download](https://huggingface.co/wjdghks950/vicuna_clip_patch) | [Download](https://huggingface.co/wjdghks950/vicuna_clip_hybrid) |
| Vicuna-13B / DINOv2 + text | [Download](https://huggingface.co/wjdghks950/vicuna_dinov2_patch) | [Download](https://huggingface.co/wjdghks950/vicuna_dinov2_hybrid) |
| Llama-3-8B-Instruct / SigLIP | [Download](https://huggingface.co/wjdghks950/llama3-8b_siglip_patch) | [Download](https://huggingface.co/wjdghks950/llama3-8b_siglip_hybrid) |
| Phi-1.5 / SigLIP | [Download](https://huggingface.co/wjdghks950/phi1_5_siglip_patch) | [Download](https://huggingface.co/wjdghks950/phi1_5_siglip_hybrid) |

```bash
hf download wjdghks950/vicuna_clip_hybrid \
  --local-dir checkpoints/vicuna_clip_hybrid
```

Download the entire checkpoint directory and load it through this repository. Vicuna checkpoints are full models; Llama-3 and Phi include LoRA and multimodal weights. Their base models are loaded automatically from `adapter_config.json`. For Llama-3, obtain access to `meta-llama/Meta-Llama-3-8B-Instruct` and run `hf auth login` first. Encoder weights download on first use; `k` is read from `config.json`.

## Data

See [data setup](docs/DATA.md) for downloads, directory layouts, and benchmark conversion commands.

- **Vicuna:** LLaVA 558K pretraining data and 665K instruction data.
- **Llama-3 / Phi:** Bunny pretraining data and Bunny-695K instruction data.

## Train

Stage 1 trains the projector and router; stage 2 performs instruction tuning. Use the same profile and `k` for both stages:

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

For another released variant, use its Hugging Face repository name as `--profile`, such as `phi1_5_siglip_patch`, and select the corresponding data. Llama-3 and Phi use LoRA instruction tuning. Additional profiles are `vicuna_clip_layer`, `vicuna_dinov2_layer`, and `vicuna_clip_dinov2_hybrid`; these have no public checkpoints. All settings are in [configs/models.json](configs/models.json).

Adjust GPU count, batch size, and accumulation for your hardware. Use a fresh output directory and a MoL stage-1 directory for `--pretrain-dir` (stock LLaVA/Bunny projectors lack router weights). Add `--dry-run` to preview commands or `--help` for options.

## Evaluate

After [preparing a benchmark](docs/DATA.md#evaluation-data):

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
  --model-path checkpoints/vicuna_clip_hybrid \
  --benchmark vstar \
  --question-file data/eval/vstar/questions.jsonl \
  --image-folder data/eval/vstar \
  --output-dir outputs/vicuna_clip_hybrid/vstar
```

Use the same command with any downloaded checkpoint. Supported benchmarks: `vstar`, `mmstar`, `hrbench4k`, `hrbench8k`, `realworldqa`, `naturalbench`, and `charxiv`. The evaluator selects the conversation template and router settings automatically, saving predictions and scores to a fresh output directory.

CharXiv reasoning needs judge grades. To enable paid API grading, set `OPENAI_API_KEY` and pass `--judge-model YOUR_JUDGE_MODEL`. See [compatibility notes](docs/COMPATIBILITY.md) for evaluation protocols, checkpoint details, and validation scope.

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

Built on [LLaVA](https://github.com/haotian-liu/LLaVA), [Bunny](https://github.com/BAAI-DCAI/Bunny), [CLIP](https://github.com/openai/CLIP), [DINOv2](https://github.com/facebookresearch/dinov2), and [SigLIP](https://github.com/google-research/big_vision). Source code uses the [Apache-2.0 license](licenses/Apache-2.0.txt); datasets and pretrained weights retain their respective licenses.
