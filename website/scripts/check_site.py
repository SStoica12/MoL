#!/usr/bin/env python3
"""Dependency-free deployment check for assets, links, and result integrity."""
import csv
import json
import re
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import unquote, urlsplit

SITE = Path(__file__).resolve().parents[1]


class Page(HTMLParser):
    def __init__(self):
        super().__init__()
        self.ids = set()
        self.links = []
        self.images = 0

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if "id" in attrs:
            assert attrs["id"] not in self.ids, f"Duplicate id: {attrs['id']}"
            self.ids.add(attrs["id"])
        for field in ["src", "href"]:
            if field in attrs:
                self.links.append(attrs[field])
        if tag == "meta" and attrs.get("property") == "og:image":
            self.links.append(attrs["content"])
        if tag == "img":
            assert attrs.get("alt"), "Image missing descriptive alt text"
            assert attrs.get("width") and attrs.get("height"), "Image missing dimensions"
            self.images += 1


def check():
    parser = Page()
    parser.feed((SITE / "index.html").read_text())
    for link in parser.links:
        url = urlsplit(link)
        if url.scheme or url.netloc:
            assert url.scheme in {"https", "mailto"}, f"Unexpected external URL: {link}"
            continue
        if url.path:
            assert not url.path.startswith("/"), f"Root-relative path breaks project Pages: {link}"
            path = SITE / unquote(url.path)
            assert path.is_file(), f"Missing asset: {link}"
        elif url.fragment:
            assert url.fragment in parser.ids, f"Missing anchor: {link}"

    data = json.loads((SITE / "assets/results.json").read_text())
    js = (SITE / "results-data.js").read_text()
    assert json.loads(js.split("window.MOL_RESULTS = ", 1)[1].rstrip(";\n")) == data
    with (SITE / "assets/results.csv").open(newline="") as file:
        csv_rows = list(csv.reader(file))
    expected = []
    for group in data["groups"].values():
        for row in group["rows"]:
            assert len(row["scores"]) == len(data["metrics"]) == 11
            assert all(0 <= score <= 100 for score in row["scores"])
            expected.append([group["label"], row["method"], *[f"{s:.2f}" for s in row["scores"]], group["source"]])
    assert csv_rows[1:] == expected, "CSV and interactive results differ"
    assert len(expected) == 16
    # Keep the headline claims synchronized with the underlying displayed scores.
    dino = data["groups"]["dinov2"]["rows"]
    assert round(dino[-1]["scores"][0] - dino[0]["scores"][0], 2) == 18.90
    assert data["groups"]["multi"]["rows"][-1]["scores"][0] == 90.34
    assert (SITE / "assets/mixture-of-layers.pdf").read_bytes().startswith(b"%PDF")
    # Optional source comparison: paper files are not required on the deployment branch.
    paper = SITE.parent / "paper/draft"
    if (paper / "sections/experiments.tex").exists():
        from refresh_content import rows
        actual = rows((paper / "sections/experiments.tex").read_text().split(r"\input{sections/multi_encoder_table}")[0])
        published = [row["scores"] for key, group in data["groups"].items() if key != "multi" for row in group["rows"]]
        assert actual == published, "Main paper scores changed: run refresh_content.py"
        assert rows((paper / "sections/multi_encoder_table.tex").read_text()) == [row["scores"] for row in data["groups"]["multi"]["rows"]]
    print(f"PASS: {len(parser.links)} links, {parser.images} figures, 16 result rows / 176 scores, CSV parity, headline claims, PDF.")


if __name__ == "__main__":
    check()
