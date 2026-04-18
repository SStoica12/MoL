import argparse
import json
import os
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import datasets as hf_datasets
import pyarrow.parquet as pq
from PIL import Image


def extract_archive(archive_file: Path, benchmark_dir: Path):
    with ZipFile(archive_file, "r") as zf:
        zf.extractall(benchmark_dir)


def find_annotation_file(benchmark_dir: Path, annotation_file_arg: str = None):
    if annotation_file_arg:
        path = Path(annotation_file_arg).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"MM-Vet annotation file not found: {path}")
        return path

    candidates = [
        benchmark_dir / "mm-vet.json",
        benchmark_dir / "mmvet.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate

    matches = sorted(path for path in benchmark_dir.rglob("mm-vet.json") if path.is_file())
    if matches:
        return matches[0]
    return None


def resolve_annotation_file(benchmark_dir: Path, annotation_file_arg: str = None):
    annotation_file = find_annotation_file(benchmark_dir, annotation_file_arg=annotation_file_arg)
    if annotation_file is not None:
        return annotation_file
    raise FileNotFoundError(
        f"Could not find mm-vet.json under {benchmark_dir}. "
        "Extract the official mm-vet.zip, pass --annotation-file, or use --hf-dataset."
    )


def resolve_image_root(benchmark_dir: Path):
    candidates = [
        benchmark_dir / "images",
        benchmark_dir / "mm-vet" / "images",
    ]
    for candidate in candidates:
        if candidate.is_dir():
            return candidate

    matches = sorted(path for path in benchmark_dir.rglob("images") if path.is_dir())
    if matches:
        return matches[0]

    raise FileNotFoundError(
        f"Could not find an images directory under {benchmark_dir}. "
        "Extract the official mm-vet.zip contents first."
    )


def normalize_capabilities(value):
    if value is None:
        return []
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    parts = [part.strip() for part in text.split(",")]
    return [part for part in parts if part]


def build_question_row(question_id, sample, benchmark_dir: Path, image_root: Path, strict: bool):
    image_name = str(sample.get("imagename", "")).strip()
    if not image_name:
        raise ValueError(f"Sample {question_id} is missing 'imagename'")

    image_path = image_root / image_name
    if not image_path.exists():
        if strict:
            raise FileNotFoundError(f"Image not found for {question_id}: {image_path}")
        rel_image_path = (image_root / image_name).relative_to(benchmark_dir).as_posix()
    else:
        rel_image_path = image_path.relative_to(benchmark_dir).as_posix()

    return {
        "question_id": str(question_id),
        "image": rel_image_path,
        "text": str(sample.get("question", "")).strip(),
        "label": str(sample.get("answer", "")).strip(),
        "question": str(sample.get("question", "")).strip(),
        "answer": str(sample.get("answer", "")).strip(),
        "capabilities": normalize_capabilities(sample.get("capability")),
        "capability": sample.get("capability"),
        "imagename": image_name,
    }


def to_pil_image(value):
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, dict):
        image_bytes = value.get("bytes")
        image_path = value.get("path")
        if image_bytes is not None:
            return Image.open(BytesIO(image_bytes))
        if image_path:
            return Image.open(image_path)
    if isinstance(value, (str, Path)):
        return Image.open(value)
    raise TypeError(f"Unsupported image payload type: {type(value).__name__}")


def load_mmvet_dataset_rows(args, benchmark_dir: Path):
    cache_dir = args.cache_dir or str(benchmark_dir / ".hf_cache")
    os.environ["HF_DATASETS_CACHE"] = cache_dir
    hf_datasets.config.HF_DATASETS_CACHE = cache_dir

    if args.parquet_dir:
        parquet_dir = Path(args.parquet_dir).expanduser().resolve()
        parquet_files = sorted(parquet_dir.rglob("*.parquet"))
        if not parquet_files:
            raise FileNotFoundError(
                f"No parquet files found under {parquet_dir}. "
                "Pass --hf-dataset or point --parquet-dir at a local MMVet snapshot."
            )
        rows = []
        for path in parquet_files:
            rows.extend(pq.read_table(path).to_pylist())
        return rows

    if args.hf_dataset:
        dataset = hf_datasets.load_dataset(
            args.hf_dataset,
            split=args.split,
            revision=args.hf_revision,
            token=args.hf_token,
            cache_dir=cache_dir,
        )
        return [dict(row) for row in dataset]

    raise ValueError("Please provide either --hf-dataset or --parquet-dir.")


def normalize_question_id(value, fallback_index, seen_ids):
    qid = str(value if value is not None else fallback_index).strip() or str(fallback_index)
    candidate = qid
    suffix = 1
    while candidate in seen_ids:
        candidate = f"{qid}_{suffix}"
        suffix += 1
    seen_ids.add(candidate)
    return candidate


def prepare_from_dataset(args, benchmark_dir: Path, output_file: Path, annotation_output_file: Path):
    dataset_rows = load_mmvet_dataset_rows(args, benchmark_dir=benchmark_dir)
    images_dir = benchmark_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    annotation_output_file.parent.mkdir(parents=True, exist_ok=True)

    annotation_by_qid = {}
    seen_ids = set()

    with open(output_file, "w") as f:
        for index, sample in enumerate(dataset_rows):
            question_id = normalize_question_id(sample.get("question_id"), index, seen_ids)
            image_name = f"{question_id}.png"
            image_path = images_dir / image_name

            image = to_pil_image(sample["image"])
            image.save(image_path, format="PNG")

            annotation_sample = {
                "imagename": image_name,
                "question": str(sample.get("question", "")).strip(),
                "answer": str(sample.get("answer", "")).strip(),
                "capability": sample.get("capability"),
                "image_source": sample.get("image_source"),
            }
            annotation_by_qid[question_id] = annotation_sample

            row = build_question_row(
                question_id=question_id,
                sample=annotation_sample,
                benchmark_dir=benchmark_dir,
                image_root=images_dir,
                strict=True,
            )
            f.write(json.dumps(row) + "\n")

    with open(annotation_output_file, "w") as f:
        json.dump(annotation_by_qid, f, indent=2, sort_keys=True)

    print(f"Exported {len(annotation_by_qid)} MM-Vet samples from dataset source")
    print(f"Annotation file: {annotation_output_file}")
    print(f"Image root: {images_dir}")
    return annotation_output_file, images_dir, len(annotation_by_qid)


def prepare_from_local_files(args, benchmark_dir: Path, output_file: Path):
    annotation_file = resolve_annotation_file(benchmark_dir, annotation_file_arg=args.annotation_file)
    image_root = resolve_image_root(benchmark_dir)

    rows = json.load(open(annotation_file, "r"))
    if not isinstance(rows, dict):
        raise ValueError(f"Expected a JSON object in {annotation_file}")

    with open(output_file, "w") as f:
        for question_id, sample in rows.items():
            row = build_question_row(
                question_id=question_id,
                sample=sample,
                benchmark_dir=benchmark_dir,
                image_root=image_root,
                strict=args.strict,
            )
            f.write(json.dumps(row) + "\n")

    print(f"Benchmark dir: {benchmark_dir}")
    print(f"Annotation file: {annotation_file}")
    print(f"Image root: {image_root}")
    print(f"Saved {len(rows)} questions to {output_file}")
    return annotation_file, image_root, len(rows)


def main():
    parser = argparse.ArgumentParser(description="Prepare MM-Vet questions.jsonl for LLaVA evaluation.")
    parser.add_argument(
        "--benchmark-dir",
        type=str,
        default="./playground/data/eval/mmvet",
        help="Directory containing the extracted MM-Vet files or where HF-exported files will be written.",
    )
    parser.add_argument(
        "--archive-file",
        type=str,
        default=None,
        help="Optional path to mm-vet.zip. If provided, it will be extracted into --benchmark-dir.",
    )
    parser.add_argument(
        "--annotation-file",
        type=str,
        default=None,
        help="Optional path to mm-vet.json. Defaults to searching under --benchmark-dir.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default=None,
        help="Optional Hugging Face dataset repo id, e.g. lmms-lab/MMVet.",
    )
    parser.add_argument(
        "--hf-revision",
        type=str,
        default="main",
        help="Revision for --hf-dataset.",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=None,
        help="Optional Hugging Face token for private/gated datasets.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional Hugging Face cache dir when using --hf-dataset.",
    )
    parser.add_argument(
        "--parquet-dir",
        type=str,
        default=None,
        help="Optional local directory containing MMVet parquet files.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        help="Dataset split to load when using --hf-dataset.",
    )
    parser.add_argument(
        "--export-annotation-file",
        type=str,
        default=None,
        help="Where to write the exported mm-vet.json when using HF/parquet data.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output jsonl file for model_vqa_loader.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Fail if an image referenced by the annotation file is missing.",
    )
    args = parser.parse_args()

    benchmark_dir = Path(args.benchmark_dir).expanduser().resolve()
    benchmark_dir.mkdir(parents=True, exist_ok=True)
    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)

    if args.archive_file:
        archive_file = Path(args.archive_file).expanduser().resolve()
        if not archive_file.exists():
            raise FileNotFoundError(f"MM-Vet archive not found: {archive_file}")
        extract_archive(archive_file, benchmark_dir)

    local_annotation_file = find_annotation_file(benchmark_dir, annotation_file_arg=args.annotation_file)
    if local_annotation_file is not None and not args.parquet_dir:
        prepare_from_local_files(args, benchmark_dir=benchmark_dir, output_file=output_file)
        return

    annotation_output_file = (
        Path(args.export_annotation_file).expanduser().resolve()
        if args.export_annotation_file
        else benchmark_dir / "mm-vet.json"
    )
    _, _, count = prepare_from_dataset(
        args,
        benchmark_dir=benchmark_dir,
        output_file=output_file,
        annotation_output_file=annotation_output_file,
    )
    print(f"Saved {count} questions to {output_file}")


if __name__ == "__main__":
    main()
