#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable

from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_TABLE_PATH = REPO_ROOT / "playground/data/eval/benchmark_overall_table.md"
DEFAULT_OUTPUT_CSV = REPO_ROOT / "playground/data/eval/benchmark_image_stats.csv"
DEFAULT_OUTPUT_MD = REPO_ROOT / "playground/data/eval/benchmark_image_stats.md"

IMAGE_EXTENSIONS = (
    "",
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp",
    ".gif",
    ".JPG",
    ".JPEG",
    ".PNG",
    ".WEBP",
    ".BMP",
    ".GIF",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    key: str
    display_name: str
    kind: str
    question_file: str | None = None
    image_root: str | None = None
    performance_column: str | None = None
    notes: str | None = None


TABLE_BENCHMARKS = [
    BenchmarkConfig(
        key="mmbench",
        display_name="MMBench",
        kind="mmbench_tsv",
        question_file="playground/data/eval/mmbench/mmbench_dev_20230712.tsv",
        performance_column="MMBench",
    ),
    BenchmarkConfig(
        key="mmmu",
        display_name="MMMU",
        kind="jsonl_image",
        question_file="playground/data/eval/MMMU/validation.jsonl",
        image_root="playground/data/eval/MMMU",
        performance_column="MMMU",
    ),
    BenchmarkConfig(
        key="mmstar",
        display_name="MMStar",
        kind="jsonl_image",
        question_file="playground/data/eval/MMStar/val.jsonl",
        image_root="playground/data/eval/MMStar",
        performance_column="MMStar",
    ),
    BenchmarkConfig(
        key="mmvet",
        display_name="MMVet",
        kind="mmvet_annotation",
        question_file="playground/data/eval/mm-vet/mm-vet.json",
        image_root="playground/data/eval/mm-vet/images",
        performance_column="MMVet",
        notes="Uses legacy local mm-vet annotation/image layout.",
    ),
    BenchmarkConfig(
        key="mmvp",
        display_name="MMVP",
        kind="jsonl_image",
        question_file="playground/data/eval/MMVP/questions.jsonl",
        image_root="playground/data/eval/MMVP",
        performance_column="MMVP",
    ),
    BenchmarkConfig(
        key="cvbench",
        display_name="CV-Bench",
        kind="jsonl_image",
        question_file="playground/data/eval/CVBench/test.jsonl",
        image_root="playground/data/eval/CVBench",
        performance_column="CV-Bench",
    ),
    BenchmarkConfig(
        key="gqa",
        display_name="GQA",
        kind="gqa_json",
        question_file="playground/data/eval/gqa/questions1.2/testdev_balanced_questions.json",
        image_root="playground/data/eval/gqa/images",
        performance_column="GQA",
    ),
    BenchmarkConfig(
        key="seed",
        display_name="SEED",
        kind="jsonl_image",
        question_file="playground/data/eval/seed_bench/llava-seed-bench.jsonl",
        image_root="playground/data/eval/seed_bench",
        performance_column="SEED",
    ),
    BenchmarkConfig(
        key="pope",
        display_name="POPE",
        kind="jsonl_image",
        question_file="playground/data/eval/pope/llava_pope_test.jsonl",
        image_root="playground/data/eval/pope/val2014",
        performance_column="POPE",
    ),
    BenchmarkConfig(
        key="textvqa",
        display_name="TextVQA",
        kind="jsonl_image",
        question_file="playground/data/eval/textvqa/llava_textvqa_val_v051_ocr.jsonl",
        image_root="playground/data/eval/textvqa/train_images",
        performance_column="TextVQA",
    ),
    BenchmarkConfig(
        key="docvqa",
        display_name="DocVQA",
        kind="jsonl_image",
        question_file="playground/data/eval/DocVQA/validation.jsonl",
        image_root="playground/data/eval/DocVQA",
        performance_column="DocVQA",
    ),
    BenchmarkConfig(
        key="mme_realworld",
        display_name="MME-RealWorld",
        kind="jsonl_image",
        question_file="playground/data/eval/MME-RealWorld/train.jsonl",
        image_root="playground/data/eval/MME-RealWorld",
        performance_column="MME-RealWorld",
    ),
    BenchmarkConfig(
        key="vqav2",
        display_name="VQAv2",
        kind="jsonl_image",
        question_file="playground/data/eval/vqav2/llava_vqav2_mscoco_test-dev2015.jsonl",
        image_root="playground/data/eval/vqav2/test2015",
        performance_column="VQAv2",
    ),
    BenchmarkConfig(
        key="hallusionbench",
        display_name="HallusionBench",
        kind="jsonl_image",
        question_file="playground/data/eval/HallusionBench/questions.jsonl",
        image_root="playground/data/eval/HallusionBench",
        performance_column="HallusionBench",
        notes="Text-only rows are counted separately and excluded from image averages.",
    ),
    BenchmarkConfig(
        key="realworldqa",
        display_name="RealWorldQA",
        kind="jsonl_image",
        question_file="playground/data/eval/RealWorldQA/test.jsonl",
        image_root="playground/data/eval/RealWorldQA",
        performance_column="RealWorldQA",
    ),
    BenchmarkConfig(
        key="chartqa",
        display_name="ChartQA",
        kind="jsonl_image",
        question_file="playground/data/eval/ChartQA/test.jsonl",
        image_root="playground/data/eval/ChartQA",
        performance_column="ChartQA",
    ),
    BenchmarkConfig(
        key="countbenchqa",
        display_name="CountBenchQA",
        kind="jsonl_image",
        question_file="playground/data/eval/CountBenchQA/test.jsonl",
        image_root="playground/data/eval/CountBenchQA",
        performance_column="CountBenchQA",
    ),
    BenchmarkConfig(
        key="hrbench4k",
        display_name="HRBench4K",
        kind="jsonl_image",
        question_file="playground/data/eval/HRBench4K/questions.jsonl",
        image_root="playground/data/eval/HRBench4K",
        performance_column="HRBench4K",
    ),
    BenchmarkConfig(
        key="hrbench8k",
        display_name="HRBench8K",
        kind="jsonl_image",
        question_file="playground/data/eval/HRBench8K/questions.jsonl",
        image_root="playground/data/eval/HRBench8K",
        performance_column="HRBench8K",
    ),
    BenchmarkConfig(
        key="vstar",
        display_name="V*Bench",
        kind="jsonl_image",
        question_file="playground/data/eval/vstar_bench/test_questions.jsonl",
        image_root="playground/data/eval/vstar_bench",
        performance_column="V*Bench",
    ),
    BenchmarkConfig(
        key="dtd",
        display_name="DTD",
        kind="jsonl_image",
        question_file="playground/data/eval/dtd/dtd/answers/test_1-10/questions.jsonl",
        image_root="playground/data/eval/dtd/dtd",
        performance_column="DTD",
    ),
]

EXTRA_BENCHMARKS = [
    BenchmarkConfig(
        key="naturalbench",
        display_name="NaturalBench",
        kind="jsonl_image",
        question_file="playground/data/eval/NaturalBench/train.jsonl",
        image_root="playground/data/eval/NaturalBench",
    ),
    BenchmarkConfig(
        key="charxiv",
        display_name="CharXiv",
        kind="jsonl_image",
        question_file="playground/data/eval/CharXiv/validation.jsonl",
        image_root="playground/data/eval/CharXiv",
        performance_column="CharXiv",
    ),
    BenchmarkConfig(
        key="llavabench",
        display_name="LLaVA-Bench",
        kind="jsonl_image",
        question_file="playground/data/eval/llava-bench-in-the-wild/questions.jsonl",
        image_root="playground/data/eval/llava-bench-in-the-wild/images",
    ),
    BenchmarkConfig(
        key="vizwiz",
        display_name="VizWiz",
        kind="jsonl_image",
        question_file="playground/data/eval/vizwiz/llava_test.jsonl",
        image_root="playground/data/eval/vizwiz/test",
    ),
    BenchmarkConfig(
        key="mme",
        display_name="MME",
        kind="jsonl_image",
        question_file="playground/data/eval/MME/llava_mme.jsonl",
        image_root="playground/data/eval/MME/MME_Benchmark_release_version",
    ),
    BenchmarkConfig(
        key="mmbench_cn",
        display_name="MMBench-CN",
        kind="mmbench_tsv",
        question_file="playground/data/eval/mmbench_cn/mmbench_dev_cn_20231003.tsv",
    ),
    BenchmarkConfig(
        key="scienceqa",
        display_name="ScienceQA",
        kind="scienceqa_json",
        question_file="playground/data/eval/scienceqa/llava_test_CQM-A.json",
        image_root="playground/data/eval/scienceqa",
    ),
]

ALL_BENCHMARKS = TABLE_BENCHMARKS + EXTRA_BENCHMARKS


@dataclass
class RunningStats:
    count: int = 0
    sum_width: float = 0.0
    sum_height: float = 0.0
    sum_pixels: float = 0.0
    sum_long_side: float = 0.0
    sum_short_side: float = 0.0

    def add(self, width: int, height: int) -> None:
        self.count += 1
        self.sum_width += width
        self.sum_height += height
        self.sum_pixels += width * height
        self.sum_long_side += max(width, height)
        self.sum_short_side += min(width, height)

    def average_width(self) -> float | None:
        return self.sum_width / self.count if self.count else None

    def average_height(self) -> float | None:
        return self.sum_height / self.count if self.count else None

    def average_pixels(self) -> float | None:
        return self.sum_pixels / self.count if self.count else None

    def average_megapixels(self) -> float | None:
        return self.average_pixels() / 1_000_000 if self.count else None

    def average_long_side(self) -> float | None:
        return self.sum_long_side / self.count if self.count else None

    def average_short_side(self) -> float | None:
        return self.sum_short_side / self.count if self.count else None


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def resolve_repo_path(path_str: str | None) -> Path | None:
    if path_str is None:
        return None
    return REPO_ROOT / path_str


def parse_markdown_table(path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(lines) < 3:
        return rows
    headers = [cell.strip() for cell in lines[0].strip("|").split("|")]
    for line in lines[2:]:
        cells = [cell.strip() for cell in line.strip("|").split("|")]
        if len(cells) != len(headers):
            continue
        rows.append(dict(zip(headers, cells)))
    return rows


def to_float(value: str | None) -> float | None:
    if value is None:
        return None
    value = value.strip()
    if not value or value == "-":
        return None
    return float(value)


def extract_score_summary(table_rows: list[dict[str, str]]) -> dict[str, dict[str, float | int | None]]:
    if not table_rows:
        return {}
    columns = [column for column in table_rows[0].keys() if column != "Model"]
    baseline_row = next((row for row in table_rows if row.get("Model") == "LLaVA v1.5 13B"), None)
    summaries: dict[str, dict[str, float | int | None]] = {}
    for column in columns:
        scores = [score for score in (to_float(row.get(column)) for row in table_rows) if score is not None]
        summaries[column] = {
            "baseline_score": to_float(baseline_row.get(column)) if baseline_row else None,
            "mean_score": mean(scores) if scores else None,
            "best_score": max(scores) if scores else None,
            "score_count": len(scores),
        }
    return summaries


def sha1_digest(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def maybe_resolve_image_path(image_root: Path | None, image_value: str | None) -> Path | None:
    if image_value is None:
        return None
    raw = str(image_value).strip()
    if not raw:
        return None
    while raw.startswith("./"):
        raw = raw[2:]

    candidate = Path(raw)
    candidates = []
    if candidate.is_absolute():
        candidates.append(candidate)
    elif image_root is not None:
        candidates.append(image_root / candidate)
    else:
        candidates.append(candidate)

    for base_candidate in candidates:
        if base_candidate.exists():
            return base_candidate
        if base_candidate.suffix:
            continue
        for suffix in IMAGE_EXTENSIONS[1:]:
            alt = Path(f"{base_candidate}{suffix}")
            if alt.exists():
                return alt
        for alt in sorted(base_candidate.parent.glob(f"{base_candidate.name}.*")):
            if alt.exists():
                return alt
    return None


def open_image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def open_base64_image_size(encoded: str) -> tuple[int, int]:
    image_bytes = base64.b64decode(encoded)
    with Image.open(io.BytesIO(image_bytes)) as image:
        return image.size


def cache_file_image_size(
    cache_key: str,
    image_path: Path,
    *,
    size_cache: dict[str, tuple[int, int]],
    result: dict[str, Any],
) -> bool:
    if cache_key in size_cache:
        return True
    try:
        size_cache[cache_key] = open_image_size(image_path)
        return True
    except Exception as exc:
        result["missing_image_count"] += 1
        if not result.get("notes"):
            result["notes"] = f"Skipped inaccessible image(s): {type(exc).__name__}"
        return False


def cache_base64_size(
    cache_key: str,
    encoded: str,
    *,
    size_cache: dict[str, tuple[int, int]],
    result: dict[str, Any],
) -> bool:
    if cache_key in size_cache:
        return True
    try:
        size_cache[cache_key] = open_base64_image_size(encoded)
        return True
    except Exception as exc:
        result["missing_image_count"] += 1
        if not result.get("notes"):
            result["notes"] = f"Skipped undecodable image(s): {type(exc).__name__}"
        return False


def rank(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    idx = 0
    while idx < len(indexed):
        end = idx + 1
        while end < len(indexed) and indexed[end][1] == indexed[idx][1]:
            end += 1
        avg_rank = (idx + 1 + end) / 2.0
        for original_idx, _ in indexed[idx:end]:
            ranks[original_idx] = avg_rank
        idx = end
    return ranks


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def spearman(xs: list[float], ys: list[float]) -> float | None:
    return pearson(rank(xs), rank(ys))


def format_float(value: float | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def safe_note_join(parts: Iterable[str | None]) -> str:
    cleaned = [part for part in parts if part]
    return "; ".join(cleaned)


def accumulate_image(
    cache_key: str,
    *,
    stats_question: RunningStats,
    stats_unique: RunningStats,
    unique_seen: set[str],
    size_cache: dict[str, tuple[int, int]],
) -> None:
    width, height = size_cache[cache_key]
    stats_question.add(width, height)
    if cache_key not in unique_seen:
        unique_seen.add(cache_key)
        stats_unique.add(width, height)


def compute_benchmark_stats(config: BenchmarkConfig) -> dict[str, Any]:
    question_file = resolve_repo_path(config.question_file)
    image_root = resolve_repo_path(config.image_root) if config.image_root else None

    result: dict[str, Any] = {
        "benchmark": config.display_name,
        "key": config.key,
        "status": "ok",
        "question_count": 0,
        "visual_question_count": 0,
        "text_only_question_count": 0,
        "missing_image_count": 0,
        "unique_image_count": 0,
        "avg_width_question": None,
        "avg_height_question": None,
        "avg_megapixels_question": None,
        "avg_long_side_question": None,
        "avg_short_side_question": None,
        "avg_width_unique": None,
        "avg_height_unique": None,
        "avg_megapixels_unique": None,
        "avg_long_side_unique": None,
        "avg_short_side_unique": None,
        "performance_column": "",
        "baseline_score": None,
        "mean_score": None,
        "best_score": None,
        "score_count": None,
        "question_file": str(question_file) if question_file else "",
        "image_root": str(image_root) if image_root else "",
        "notes": config.notes or "",
    }

    if question_file is None or not question_file.exists():
        result["status"] = "missing_question_file"
        return result

    stats_question = RunningStats()
    stats_unique = RunningStats()
    unique_seen: set[str] = set()
    size_cache: dict[str, tuple[int, int]] = {}

    if config.kind == "mmbench_tsv":
        with question_file.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                result["question_count"] += 1
                encoded = str(row.get("image", "")).strip()
                if not encoded:
                    result["text_only_question_count"] += 1
                    continue
                cache_key = f"base64:{sha1_digest(encoded)}"
                if not cache_base64_size(cache_key, encoded, size_cache=size_cache, result=result):
                    continue
                result["visual_question_count"] += 1
                accumulate_image(
                    cache_key,
                    stats_question=stats_question,
                    stats_unique=stats_unique,
                    unique_seen=unique_seen,
                    size_cache=size_cache,
                )
    elif config.kind == "gqa_json":
        payload = json.loads(question_file.read_text(encoding="utf-8"))
        for row in payload.values():
            result["question_count"] += 1
            image_path = maybe_resolve_image_path(image_root, row.get("imageId"))
            if image_path is None:
                result["missing_image_count"] += 1
                continue
            cache_key = str(image_path.resolve())
            if not cache_file_image_size(cache_key, image_path, size_cache=size_cache, result=result):
                continue
            result["visual_question_count"] += 1
            accumulate_image(
                cache_key,
                stats_question=stats_question,
                stats_unique=stats_unique,
                unique_seen=unique_seen,
                size_cache=size_cache,
            )
    elif config.kind == "mmvet_annotation":
        payload = json.loads(question_file.read_text(encoding="utf-8"))
        for row in payload.values():
            result["question_count"] += 1
            image_path = maybe_resolve_image_path(image_root, row.get("imagename"))
            if image_path is None:
                result["missing_image_count"] += 1
                continue
            cache_key = str(image_path.resolve())
            if not cache_file_image_size(cache_key, image_path, size_cache=size_cache, result=result):
                continue
            result["visual_question_count"] += 1
            accumulate_image(
                cache_key,
                stats_question=stats_question,
                stats_unique=stats_unique,
                unique_seen=unique_seen,
                size_cache=size_cache,
            )
    elif config.kind == "scienceqa_json":
        payload = json.loads(question_file.read_text(encoding="utf-8"))
        for row in payload:
            result["question_count"] += 1
            image_value = row.get("image")
            if not image_value:
                result["text_only_question_count"] += 1
                continue
            image_path = maybe_resolve_image_path(image_root, image_value)
            if image_path is None:
                result["missing_image_count"] += 1
                continue
            cache_key = str(image_path.resolve())
            if not cache_file_image_size(cache_key, image_path, size_cache=size_cache, result=result):
                continue
            result["visual_question_count"] += 1
            accumulate_image(
                cache_key,
                stats_question=stats_question,
                stats_unique=stats_unique,
                unique_seen=unique_seen,
                size_cache=size_cache,
            )
    else:
        for row in read_jsonl(question_file):
            result["question_count"] += 1
            image_value = row.get("image")
            if image_value is None or str(image_value).strip() == "":
                result["text_only_question_count"] += 1
                continue
            image_path = maybe_resolve_image_path(image_root, str(image_value))
            if image_path is None:
                result["missing_image_count"] += 1
                continue
            cache_key = str(image_path.resolve())
            if not cache_file_image_size(cache_key, image_path, size_cache=size_cache, result=result):
                continue
            result["visual_question_count"] += 1
            accumulate_image(
                cache_key,
                stats_question=stats_question,
                stats_unique=stats_unique,
                unique_seen=unique_seen,
                size_cache=size_cache,
            )

    result["unique_image_count"] = stats_unique.count
    result["avg_width_question"] = stats_question.average_width()
    result["avg_height_question"] = stats_question.average_height()
    result["avg_megapixels_question"] = stats_question.average_megapixels()
    result["avg_long_side_question"] = stats_question.average_long_side()
    result["avg_short_side_question"] = stats_question.average_short_side()
    result["avg_width_unique"] = stats_unique.average_width()
    result["avg_height_unique"] = stats_unique.average_height()
    result["avg_megapixels_unique"] = stats_unique.average_megapixels()
    result["avg_long_side_unique"] = stats_unique.average_long_side()
    result["avg_short_side_unique"] = stats_unique.average_short_side()
    if result["visual_question_count"] == 0 and result["status"] == "ok":
        if result["missing_image_count"] > 0:
            result["status"] = "missing_image_assets"
        else:
            result["status"] = "no_visual_rows"
    return result


def add_performance_columns(
    rows: list[dict[str, Any]],
    score_summary: dict[str, dict[str, float | int | None]],
) -> None:
    perf_lookup = {
        config.key: config.performance_column
        for config in ALL_BENCHMARKS
        if config.performance_column
    }
    for row in rows:
        column = perf_lookup.get(row["key"])
        summary = score_summary.get(column, {})
        row["performance_column"] = column or ""
        row["baseline_score"] = summary.get("baseline_score")
        row["mean_score"] = summary.get("mean_score")
        row["best_score"] = summary.get("best_score")
        row["score_count"] = summary.get("score_count")


def build_correlation_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    measured = [
        row
        for row in rows
        if row["performance_column"]
        and row["status"] == "ok"
        and row["avg_megapixels_question"] is not None
        and row["avg_megapixels_question"] > 0
    ]

    def paired(metric: str) -> tuple[list[float], list[float]]:
        pairs = [
            (math.log10(row["avg_megapixels_question"]), row[metric])
            for row in measured
            if row.get(metric) is not None
        ]
        return [pair[0] for pair in pairs], [pair[1] for pair in pairs]

    baseline_x, baseline_y = paired("baseline_score")
    mean_x, mean_y = paired("mean_score")
    best_x, best_y = paired("best_score")
    return {
        "benchmark_count": len(measured),
        "baseline_score_vs_log10_megapixels": {
            "pearson": pearson(baseline_x, baseline_y),
            "spearman": spearman(baseline_x, baseline_y),
            "count": len(baseline_x),
        },
        "mean_score_vs_log10_megapixels": {
            "pearson": pearson(mean_x, mean_y),
            "spearman": spearman(mean_x, mean_y),
            "count": len(mean_x),
        },
        "best_score_vs_log10_megapixels": {
            "pearson": pearson(best_x, best_y),
            "spearman": spearman(best_x, best_y),
            "count": len(best_x),
        },
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "benchmark",
        "key",
        "status",
        "question_count",
        "visual_question_count",
        "text_only_question_count",
        "missing_image_count",
        "unique_image_count",
        "avg_width_question",
        "avg_height_question",
        "avg_megapixels_question",
        "avg_long_side_question",
        "avg_short_side_question",
        "avg_width_unique",
        "avg_height_unique",
        "avg_megapixels_unique",
        "avg_long_side_unique",
        "avg_short_side_unique",
        "performance_column",
        "baseline_score",
        "mean_score",
        "best_score",
        "score_count",
        "question_file",
        "image_root",
        "notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_markdown(rows: list[dict[str, Any]], correlations: dict[str, Any]) -> str:
    lines = ["# Benchmark Image Stats", ""]
    lines.extend(
        [
            "| Benchmark | Status | Q-Weighted Avg WxH | Q-Weighted MP | Unique Avg WxH | Unique MP | Visual Qs | Unique Images | Baseline | Mean |",
            "| --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        q_dims = ""
        if row["avg_width_question"] is not None and row["avg_height_question"] is not None:
            q_dims = f"{row['avg_width_question']:.0f} x {row['avg_height_question']:.0f}"
        u_dims = ""
        if row["avg_width_unique"] is not None and row["avg_height_unique"] is not None:
            u_dims = f"{row['avg_width_unique']:.0f} x {row['avg_height_unique']:.0f}"
        lines.append(
            "| {benchmark} | {status} | {q_dims} | {q_mp} | {u_dims} | {u_mp} | {visual_q} | {unique_imgs} | {baseline} | {mean_score} |".format(
                benchmark=row["benchmark"],
                status=row["status"],
                q_dims=q_dims,
                q_mp=format_float(row["avg_megapixels_question"]),
                u_dims=u_dims,
                u_mp=format_float(row["avg_megapixels_unique"]),
                visual_q=row["visual_question_count"],
                unique_imgs=row["unique_image_count"],
                baseline=format_float(row["baseline_score"]),
                mean_score=format_float(row["mean_score"]),
            )
        )
    lines.append("")
    lines.append("## Correlations")
    for label, summary in correlations.items():
        if not isinstance(summary, dict):
            continue
        lines.append(
            f"- `{label}`: n={summary.get('count', 0)}, "
            f"pearson={format_float(summary.get('pearson'), 3)}, "
            f"spearman={format_float(summary.get('spearman'), 3)}"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- `Q-Weighted` averages follow the benchmark question distribution, so repeated use of the same image is counted multiple times.")
    lines.append("- `Unique` averages de-duplicate by image identity or base64 payload.")
    lines.append("- HallusionBench includes text-only rows; they are excluded from image-dimension averages and counted separately.")
    lines.append("- Raw benchmark resolution is not the exact inference resolution. LLaVA still preprocesses images through the CLIP image processor in `llava/mm_utils.py`.")
    missing = [row for row in rows if row["status"] != "ok"]
    if missing:
        lines.append("- Missing or unsupported local benchmarks:")
        for row in missing:
            note = safe_note_join(
                [
                    row["status"],
                    row["notes"],
                    row["question_file"] if row["question_file"] else None,
                ]
            )
            lines.append(f"  - {row['benchmark']}: {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute benchmark image-dimension statistics and join them to the eval score table.")
    parser.add_argument(
        "--score-table",
        type=Path,
        default=DEFAULT_TABLE_PATH,
        help="Markdown performance table to join against.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=DEFAULT_OUTPUT_CSV,
        help="CSV output path.",
    )
    parser.add_argument(
        "--output-md",
        type=Path,
        default=DEFAULT_OUTPUT_MD,
        help="Markdown report path.",
    )
    parser.add_argument(
        "--table-only",
        action="store_true",
        help="Restrict the scan to benchmarks that appear in benchmark_overall_table.md.",
    )
    args = parser.parse_args()

    configs = TABLE_BENCHMARKS if args.table_only else ALL_BENCHMARKS
    rows = [compute_benchmark_stats(config) for config in configs]
    score_summary = extract_score_summary(parse_markdown_table(args.score_table))
    add_performance_columns(rows, score_summary)
    rows.sort(
        key=lambda row: (
            0 if row["performance_column"] else 1,
            row["status"] != "ok",
            -(row["avg_megapixels_question"] or -1),
            row["benchmark"],
        )
    )

    correlations = build_correlation_summary(rows)
    write_csv(args.output_csv, rows)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(build_markdown(rows, correlations), encoding="utf-8")

    print(f"Wrote CSV report to {args.output_csv}")
    print(f"Wrote Markdown report to {args.output_md}")
    for row in rows:
        q_dims = ""
        if row["avg_width_question"] is not None and row["avg_height_question"] is not None:
            q_dims = f"{row['avg_width_question']:.0f}x{row['avg_height_question']:.0f}"
        print(
            f"{row['benchmark']:16s} "
            f"status={row['status']:>20s} "
            f"q_avg={q_dims or '-':>14s} "
            f"q_mp={format_float(row['avg_megapixels_question']) or '-':>6s} "
            f"baseline={format_float(row['baseline_score']) or '-':>6s}"
        )


if __name__ == "__main__":
    main()
