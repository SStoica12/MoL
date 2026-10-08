# Data setup

Run all commands from the repository root in the `mol` environment. Downloads are user-managed: the training launcher does not download datasets. Dataset image files must be extracted before training.

## Vicuna training data

### Caption alignment: LLaVA 558K

Download the annotation and image archive from [LLaVA-Pretrain](https://huggingface.co/datasets/liuhaotian/LLaVA-Pretrain):

```bash
hf download liuhaotian/LLaVA-Pretrain --repo-type dataset \
  --include blip_laion_cc_sbu_558k.json images.zip \
  --local-dir data/pretrain
unzip data/pretrain/images.zip -d data/pretrain
```

The final layout should be `data/pretrain/blip_laion_cc_sbu_558k.json` and `data/pretrain/images/<image filename>`. If your extraction adds an extra `images/` directory, move it or adjust `--image-folder` so each annotation's `image` field resolves correctly.

### Instruction tuning: LLaVA 665K

```bash
hf download liuhaotian/LLaVA-Instruct-150K --repo-type dataset \
  --include llava_v1_5_mix665k.json --local-dir data/train
```

Download images following [LLaVA's training data instructions](https://github.com/haotian-liu/LLaVA#visual-instruction-tuning): [COCO train2017](https://images.cocodataset.org/zips/train2017.zip), [GQA images](https://downloads.cs.stanford.edu/nlp/data/gqa/images.zip), OCR-VQA (the upstream download script), [TextVQA](https://dl.fbaipublicfiles.com/textvqa/images/train_val_images.zip), and Visual Genome [part 1](https://cs.stanford.edu/people/rak248/VG_100K_2/images.zip) / [part 2](https://cs.stanford.edu/people/rak248/VG_100K_2/images2.zip). OCR-VQA images must use the `.jpg` filenames expected by the annotations.

Arrange the image root as:

```text
data/train/
├── llava_v1_5_mix665k.json
├── coco/train2017/
├── gqa/images/
├── ocr_vqa/images/
├── textvqa/train_images/
└── vg/
    ├── VG_100K/
    └── VG_100K_2/
```

Use `--image-folder data/train`. Both single-encoder Vicuna profiles and the dual-encoder Vicuna profile consume these same annotations and images.

## Bunny training data

For the Llama-3/SigLIP and Phi-1.5/SigLIP profiles, use [Bunny's data instructions](https://github.com/BAAI-DCAI/Bunny#train). Its pretraining set contains 2M caption-image pairs sampled from a LAION coreset; the instruction data here is Bunny-695K. The training launcher selects the matching language model and conversation template for each profile.

- [Bunny-v1_1-data](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_1-data/tree/main/pretrain): pretraining annotations and archives.
- [Bunny-v1_0-data](https://huggingface.co/datasets/BoyaWu10/Bunny-v1_0-data/tree/main): Bunny-695K annotations and image archives.

To download the pretraining subdirectory and instruction assets:

```bash
hf download BoyaWu10/Bunny-v1_1-data --repo-type dataset \
  --include 'pretrain/*' --local-dir data/bunny
hf download BoyaWu10/Bunny-v1_0-data --repo-type dataset \
  --include 'finetune/*' --local-dir data/bunny-695k
cat data/bunny/pretrain/images.tar.gz.part-* | tar -xz -C data/bunny/pretrain
cat data/bunny-695k/finetune/images.tar.gz.part-* | tar -xz -C data/bunny-695k/finetune
```

Use `data/bunny/pretrain/bunny_pretrain_laion_2m.json` and `data/bunny/pretrain/images` for stage 1; use `data/bunny-695k/finetune/bunny_695k.json` and `data/bunny-695k/finetune/images` for stage 2. Check the annotation paths after extraction using the snippet below and adjust the image root if the archive layout differs. Do not substitute the larger Bunny-v1.1 instruction mixtures when intending to use Bunny-695K. Training from these datasets follows the two-stage MoL recipe in the README; it does not load a stock Bunny projector into MoL's strict router loader.

## Annotation format and image checks

Training expects a JSON **array**, not JSONL. For example:

```json
[
  {
    "id": "example-1",
    "image": "coco/train2017/000000000009.jpg",
    "conversations": [
      {"from": "human", "value": "<image>\nWhat is on the table?"},
      {"from": "gpt", "value": "A plate and a cup."}
    ]
  }
]
```

Caption pretraining uses one image and one human/assistant exchange per sample. Instruction tuning can contain multiple turns. Keep the `<image>` marker in the image-bearing conversation. Text-only instruction examples can omit `image`.

Check every referenced image before starting a long training run (adjust these two paths):

```bash
python - <<'PY'
import json
from pathlib import Path
annotations = Path('data/train/llava_v1_5_mix665k.json')
image_root = Path('data/train')
rows = json.loads(annotations.read_text())
assert isinstance(rows, list), 'Expected a JSON list'
missing = [r['image'] for r in rows if r.get('image') and not (image_root / r['image']).is_file()]
print(f'{len(rows)} examples; {len(missing)} missing images')
print(missing[:10])
assert not missing, 'Fix image extraction or the image root before training'
PY
```

## Evaluation data

The converters download public source data and emit JSONL with `question_id`, `image`, `text`, and benchmark-specific labels. Use the same directory for the converter's benchmark root and the evaluator's `--image-folder`.

```bash
# V*Bench
python scripts/convert_vstar_for_eval.py \
  --hf-dataset craigwu/vstar_bench --download-dir data/eval/vstar \
  --output-file data/eval/vstar/questions.jsonl --strict

# MMStar
python scripts/convert_mmstar_for_eval.py \
  --benchmark-dir data/eval/mmstar --output-file data/eval/mmstar/questions.jsonl

# HRBench, both image resolutions
python scripts/convert_hrbench_for_eval.py --split hrbench_4k \
  --benchmark-dir data/eval/hrbench4k --output-file data/eval/hrbench4k/questions.jsonl --strict
python scripts/convert_hrbench_for_eval.py --split hrbench_8k \
  --benchmark-dir data/eval/hrbench8k --output-file data/eval/hrbench8k/questions.jsonl --strict

# RealWorldQA
python scripts/convert_realworldqa_for_eval.py --hf-dataset lmms-lab/RealWorldQA \
  --benchmark-dir data/eval/realworldqa --output-file data/eval/realworldqa/questions.jsonl

# NaturalBench
python scripts/convert_naturalbench_for_eval.py \
  --benchmark-dir data/eval/naturalbench --output-file data/eval/naturalbench/questions.jsonl

# CharXiv validation
python scripts/convert_charxiv_for_eval.py --split validation \
  --benchmark-dir data/eval/charxiv --output-file data/eval/charxiv/questions.jsonl
```

Sources: [V*Bench](https://huggingface.co/datasets/craigwu/vstar_bench), [MMStar](https://huggingface.co/datasets/Lin-Chen/MMStar), [HR-Bench](https://huggingface.co/datasets/DreamMr/HR-Bench), [RealWorldQA](https://huggingface.co/datasets/lmms-lab/RealWorldQA), [NaturalBench](https://huggingface.co/datasets/BaiqiL/NaturalBench), [CharXiv](https://huggingface.co/datasets/princeton-nlp/CharXiv).

Every converter accepts `--hf-revision` to pin the source revision; record that revision for comparisons. Run a converter with `--help` to use already downloaded JSON/image folders, Parquet files, or HRBench TSVs. Conversion does not run models or judges. Preserve NaturalBench question/image groups for its group-based metrics. Public V*Bench data does not necessarily contain the extra OCR/GPT4V-hard categories used by older local runs; the converter reports only categories actually present.
