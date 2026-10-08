#!/usr/bin/env python3
"""Run the existing MoL inference and scoring entry points on one benchmark."""
import argparse
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
SCORERS = {name: name for name in ("vstar", "mmstar", "realworldqa", "naturalbench", "charxiv")}
SCORERS.update(hrbench4k="hrbench", hrbench8k="hrbench")


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-path", type=Path, required=True, help="Downloaded MoL checkpoint directory")
    p.add_argument("--model-base", help="Override adapter base with a local path or Hub ID")
    p.add_argument("--conv-mode", help="Optional conversation template override")
    p.add_argument("--benchmark", choices=SCORERS, required=True)
    p.add_argument("--question-file", type=Path, required=True)
    p.add_argument("--image-folder", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--k", type=int, help="Optional evaluation override; normally use checkpoint config")
    p.add_argument("--max-new-tokens", type=int)
    p.add_argument("--judge-model", help="CharXiv only: explicitly enable external API reasoning grading")
    p.add_argument("--dry-run", action="store_true")
    return p


def commands(args):
    folder = args.model_path.expanduser().resolve()
    cfg = json.loads((folder / "config.json").read_text())
    adapter_path = folder / "adapter_config.json"
    adapter = json.loads(adapter_path.read_text()) if adapter_path.is_file() else {}
    base = args.model_base or adapter.get("base_model_name_or_path")
    identity = " ".join(str(x) for x in (
        cfg.get("model_type"), cfg.get("architectures"), cfg.get("_name_or_path"),
        adapter.get("base_model_name_or_path"), base,
    )).lower()
    if "phi" in identity:
        raise ValueError("Phi loading is not connected in this snapshot; see docs/COMPATIBILITY.md.")
    if adapter and not base:
        raise ValueError("This adapter needs --model-base or base_model_name_or_path in adapter_config.json.")
    conv = args.conv_mode or ("llama" if "llama-3" in identity or "llama3" in identity else "vicuna_v1")
    if args.judge_model and args.benchmark != "charxiv":
        raise ValueError("--judge-model is only used for CharXiv.")
    if args.k is not None and args.k <= 0:
        raise ValueError("--k must be positive.")
    out = args.output_dir.resolve()
    infer = [sys.executable, "-m", "llava.eval.model_vqa_loader",
             "--model-path", str(folder), "--use_router", "--conv-mode", conv,
             "--question-file", str(args.question_file.resolve()),
             "--image-folder", str(args.image_folder.resolve()),
             "--answers-file", str(out / "answers.jsonl"), "--benchmark", args.benchmark,
             "--temperature", "0", "--num_beams", "1", "--get_masks", "0",
             "--max_new_tokens", str(args.max_new_tokens or (256 if args.benchmark == "charxiv" else 128)),
             "--output_path_vis", str(out / "router"),
             "--output_grouped_questions", str(out / "grouped_questions.json")]
    if base:
        infer.extend(["--model-base", base])
    if args.k is not None:
        infer.extend(["--num_layers_to_choose", str(args.k)])
    score = [sys.executable, "-m", "llava.eval.eval_" + SCORERS[args.benchmark],
             "--question-file", str(args.question_file.resolve()),
             "--result-file", str(out / "answers.jsonl"),
             "--output-file", str(out / "metrics.json"),
             "--detail-file", str(out / "details.jsonl")]
    if args.benchmark == "charxiv":
        score.extend(["--grade-file", str(out / "grades.json")])
        if args.judge_model:
            score.extend(["--judge-model", args.judge_model])
    return [infer, score]


def main():
    p = parser()
    args = p.parse_args()
    try:
        cmds = commands(args)
        if not args.dry_run:
            if not args.question_file.is_file() or not args.image_folder.is_dir():
                raise ValueError("--question-file and --image-folder must exist; see docs/DATA.md.")
            if args.output_dir.exists() and any(args.output_dir.iterdir()):
                raise ValueError("Choose a new --output-dir to preserve previous predictions.")
            if args.judge_model and not os.environ.get("OPENAI_API_KEY"):
                raise ValueError("Set OPENAI_API_KEY before enabling CharXiv API grading.")
            args.output_dir.mkdir(parents=True, exist_ok=True)
        env = os.environ.copy()
        env["ROUTER_DISTRIBUTION_DIR"] = str(args.output_dir.resolve() / "router")
        for cmd in cmds:
            print(shlex.join(cmd), flush=True)
            if not args.dry_run:
                subprocess.run(cmd, check=True, cwd=ROOT, env=env)
    except (ValueError, OSError) as exc:
        p.error(str(exc))


if __name__ == "__main__":
    main()
