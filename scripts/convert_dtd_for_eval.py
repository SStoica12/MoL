import argparse
import json
from pathlib import Path


def read_split_file(split_file: Path):
    with open(split_file, "r") as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines


def discover_classes(images_dir: Path):
    classes = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
    if not classes:
        raise FileNotFoundError(f"No class folders found under {images_dir}")
    return classes


def format_prompt(classes):
    class_list = ", ".join(classes)
    return (
        "What is the texture category of this image?\n"
        f"Choose exactly one class from this list: {class_list}\n"
        "Answer with only the class name."
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert DTD split files to LLaVA JSONL format."
    )
    parser.add_argument(
        "--dtd-root",
        type=str,
        required=True,
        help="Path to DTD root containing images/ and labels/ directories.",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="DTD split type.",
    )
    parser.add_argument(
        "--split-ids",
        nargs="+",
        type=int,
        default=[1],
        help="Split IDs to include (1..10).",
    )
    parser.add_argument(
        "--all-splits",
        action="store_true",
        help="Shortcut for --split-ids 1 2 3 4 5 6 7 8 9 10.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        required=True,
        help="Output JSONL path.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dtd_root = Path(args.dtd_root).expanduser().resolve()
    images_dir = dtd_root / "images"
    labels_dir = dtd_root / "labels"

    if not images_dir.exists():
        raise FileNotFoundError(f"images directory not found: {images_dir}")
    if not labels_dir.exists():
        raise FileNotFoundError(f"labels directory not found: {labels_dir}")

    split_ids = list(range(1, 11)) if args.all_splits else args.split_ids
    for sid in split_ids:
        if sid < 1 or sid > 10:
            raise ValueError(f"Split id must be in [1, 10], got {sid}")

    classes = discover_classes(images_dir)
    prompt = format_prompt(classes)

    records = []
    per_split_counts = {}
    for sid in split_ids:
        split_file = labels_dir / f"{args.split}{sid}.txt"
        if not split_file.exists():
            raise FileNotFoundError(f"Split file not found: {split_file}")

        rel_paths = read_split_file(split_file)
        count = 0
        for rel_path in rel_paths:
            rel_path = rel_path.strip()
            # rel_path example: banded/banded_0002.jpg
            label = rel_path.split("/", 1)[0]
            image_rel = f"images/{rel_path}"
            image_abs = dtd_root / image_rel
            if not image_abs.exists():
                raise FileNotFoundError(f"Image listed in split not found: {image_abs}")

            qid = f"{args.split}{sid}:{rel_path}"
            records.append(
                {
                    "question_id": qid,
                    "image": image_rel,
                    "text": prompt,
                    "label": label,
                    "split": args.split,
                    "split_id": sid,
                }
            )
            count += 1
        per_split_counts[sid] = count

    output_file = Path(args.output_file).expanduser().resolve()
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"Saved {len(records)} samples to {output_file}")
    print(f"Classes ({len(classes)}): {', '.join(classes)}")
    for sid in split_ids:
        print(f"{args.split}{sid}: {per_split_counts[sid]} samples")


if __name__ == "__main__":
    main()
