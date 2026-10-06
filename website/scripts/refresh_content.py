#!/usr/bin/env python3
"""Refresh paper-derived assets and scores. Requires PyMuPDF and Pillow."""
import csv
import hashlib
import io
import json
import re
import shutil
from pathlib import Path

SITE = Path(__file__).resolve().parents[1]
ROOT = SITE.parent
PAPER = ROOT / "paper/draft"
METRICS = ["V*", "GPT4V-hard", "OCR", "Direct attr.", "Rel. pos.", "MMStar",
           "HRBench8K", "HRBench4K", "RealWorldQA", "NaturalBench", "CharXiv"]


def rows(source):
    """Read displayed numeric rows, excluding comments and computed deltas."""
    source = re.sub(r"(?<!\\)%[^\n]*", "", source)
    found = []
    for line in source.splitlines():
        if "&" not in line or "\\\\" not in line or "Delta" in line:
            continue
        cells = line.split("&")
        scores = [re.findall(r"(?<![\d.])\d+\.\d+", cell) for cell in cells[1:]]
        if len(scores) == 11 and all(len(values) == 1 for values in scores):
            found.append([float(values[0]) for values in scores])
    return found


def refresh_scores():
    source = (PAPER / "sections/experiments.tex").read_text()
    main = rows(source.split(r"\input{sections/multi_encoder_table}")[0])
    multi = rows((PAPER / "sections/multi_encoder_table.tex").read_text())
    if len(main) != 14 or len(multi) != 2:
        raise ValueError("Paper table layout changed; review extraction before publishing.")
    groups = [
        ("clip", "Vicuna-13B + CLIP", main[:4], ["Baseline", "MoL layer", "MoL patch", "MoL hybrid"]),
        ("dinov2", "Vicuna-13B + DINOv2 w/ Txt", main[4:8], ["Baseline", "MoL layer", "MoL patch", "MoL hybrid"]),
        ("llama", "Llama-3-8B + SigLIP", main[8:11], ["Baseline", "MoL patch", "MoL hybrid"]),
        ("phi", "Phi-1.5-1.3B + SigLIP", main[11:], ["Baseline", "MoL patch", "MoL hybrid"]),
        ("multi", "Multi-encoder · Vicuna-13B", multi, ["Interleaved-MoF", "MoL hybrid"]),
    ]
    data = {"metrics": METRICS, "groups": {}}
    csv_file = io.StringIO()
    writer = csv.writer(csv_file, lineterminator="\n")
    writer.writerow(["Backbone", "Method", *METRICS, "Source"])
    for key, label, values, methods in groups:
        source_file = "multi_encoder_table.tex" if key == "multi" else "experiments.tex (main backbone table)"
        data["groups"][key] = {"label": label, "rows": [{"method": m, "scores": s} for m, s in zip(methods, values)], "source": source_file}
        for method, scores in zip(methods, values):
            writer.writerow([label, method, *[f"{s:.2f}" for s in scores], source_file])
    (SITE / "results-data.js").write_text("// Generated from the paper by scripts/refresh_content.py; do not edit.\nwindow.MOL_RESULTS = " + json.dumps(data, indent=2) + ";\n")
    (SITE / "assets/results.csv").write_text(csv_file.getvalue())
    (SITE / "assets/results.json").write_text(json.dumps(data, indent=2) + "\n")


def refresh_figures():
    import pymupdf
    from PIL import Image

    figures = ["main_figure", "case_study_attention_map", "case_study_attention_map_dinov2",
               "layer_sampling_by_category", "vstar_mmstar_gqa_layer_sampling_rf_background_dinov2"]
    for name in figures:
        with pymupdf.open(PAPER / f"figures/{name}.pdf") as doc:
            page = doc[0]
            pix = page.get_pixmap(matrix=pymupdf.Matrix(2400 / page.rect.width, 2400 / page.rect.width), alpha=False)
            Image.frombytes("RGB", (pix.width, pix.height), pix.samples).save(SITE / f"assets/{name}.webp", "WEBP", quality=88)
    shutil.copy2(PAPER / "neurips_2024.pdf", SITE / "assets/mixture-of-layers.pdf")


def provenance():
    paths = ["neurips_2024.tex", "neurips_2024.pdf", "sections/abstract.tex", "sections/introduction.tex",
             "sections/method.tex", "sections/experiments.tex", "sections/multi_encoder_table.tex"]
    paths += [str(p.relative_to(PAPER)) for p in sorted((PAPER / "figures").glob("*.pdf"))]
    manifest = {str(Path("paper/draft") / p): hashlib.sha256((PAPER / p).read_bytes()).hexdigest() for p in paths}
    (SITE / "source-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    refresh_scores()
    refresh_figures()
    provenance()
    print("Refreshed five backbone comparisons, CSV/JSON, five paper figures, PDF, and source hashes.")
