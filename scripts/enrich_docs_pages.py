"""Insert generated Qortex evidence blocks where the figure adds evidence.

The generated blocks are deliberately marked so they can be refreshed. Pages
not listed in ``PAGE_CARDS`` have generated evidence removed; a figure should
earn its place by showing a result that the page actually teaches.

Run after:
    python scripts/generate_docs_examples.py
"""

from __future__ import annotations

import json
import re
from pathlib import Path


DOCS = Path("docs")
INDEX_PATH = DOCS / "assets" / "results" / "docs-evidence-index.json"
START = "<!-- qortex-evidence:start -->"
END = "<!-- qortex-evidence:end -->"


def _rel(page: Path, docs_asset_path: str) -> str:
    _ = page
    return "/Qortex/" + docs_asset_path.removeprefix("docs/").lstrip("/")


PAGE_CARDS: dict[str, str] = {
    "index.md": "visualization",
    "getting-started/index.md": "dataset",
    "getting-started/quickstart.md": "minimum",
    "getting-started/first-visual-audit.md": "visualization",
    "dataset/index.md": "dataset",
    "dataset/inspect.md": "dataset",
    "dataset/metadata.md": "metadata",
    "dataset/snapshots.md": "dataset",
    "download/index.md": "minimum",
    "download/plan.md": "minimum",
    "download/metadata-only.md": "minimum",
    "download/selective-download.md": "minimum",
    "download/cache.md": "content",
    "readiness/index.md": "readiness",
    "readiness/can-train.md": "readiness",
    "readiness/doctor.md": "readiness",
    "readiness/first-batch.md": "minimum",
    "readiness/minimum.md": "minimum",
    "readiness/label-readiness.md": "events",
    "modalities/behavioral/events-tsv.md": "events",
    "modalities/behavioral/labels-and-trial-types.md": "events",
    "modalities/eeg/events.md": "events",
    "modalities/meg/events.md": "events",
    "modalities/mri/fmri-bold.md": "visualization",
    "visualization/index.md": "visualization",
    "visualization/fmri-qc.md": "visualization",
    "visualization/local-viewer.md": "visualization",
    "visualization/visual-audit.md": "visualization",
    "visualization/visualize-openneuro.md": "visualization",
    "conversion/index.md": "conversion",
    "conversion/splits.md": "conversion",
    "conversion/pipeline.md": "conversion",
    "artifacts/index.md": "conversion",
    "artifacts/manifest.md": "conversion",
    "artifacts/ml-bridge.md": "conversion",
    "neuroai/index.md": "neuroai",
    "neuroai/pipeline.md": "neuroai",
    "neuroai/models.md": "model_sources",
    "neuroai/sources.md": "neuroai",
    "neuroai/outputs.md": "neuroai",
    "tutorials/t05-mri-dementia-baseline.md": "metadata",
    "tutorials/t06-mri-age-sex-qc.md": "metadata",
    "tutorials/t07-fmri-design-readiness.md": "events",
    "tutorials/t08-brain-tumour-segmentation.md": "model_sources",
    "troubleshooting/downloads.md": "content",
    "troubleshooting/openneuro.md": "content",
    "troubleshooting/labels.md": "events",
}


def _block(page: Path, card: dict[str, str]) -> str:
    image = _rel(page, card["image"])
    result = card["result"]
    result_label = Path(result).name
    result_rel = _rel(page, result.removeprefix("docs/"))
    code = card["code"].strip()
    lang = "bash" if code.startswith("qortex ") else "python"
    return f"""
{START}

## Evidence

<figure class="tq-figure">
  <img src="{image}" alt="{card['alt']}">
  <figcaption>{card['caption']}</figcaption>
</figure>

```{lang}
{code}
```

Result artifact: [{result_label}]({result_rel})

{END}
""".strip()


def _insert_or_replace(text: str, block: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"\n?{re.escape(START)}.*?{re.escape(END)}\n?",
        flags=re.DOTALL,
    )
    if pattern.search(text):
        return pattern.sub("\n\n" + block + "\n", text).rstrip() + "\n", False

    related_match = re.search(r"\n## Related\b", text)
    if related_match:
        idx = related_match.start()
        return (text[:idx].rstrip() + "\n\n" + block + "\n" + text[idx:]).rstrip() + "\n", True
    return text.rstrip() + "\n\n" + block + "\n", True


def _remove_block(text: str) -> tuple[str, bool]:
    pattern = re.compile(
        rf"\n?{re.escape(START)}.*?{re.escape(END)}\n?",
        flags=re.DOTALL,
    )
    new_text = pattern.sub("\n", text).rstrip() + "\n"
    return new_text, new_text != text


def main() -> None:
    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    cards = index["cards"]
    pages = sorted(
        p for p in DOCS.rglob("*.md")
        if "assets" not in p.parts
    )
    changed = 0
    inserted = 0
    removed = 0
    for page in pages:
        rel = page.relative_to(DOCS).as_posix()
        text = page.read_text(encoding="utf-8")
        key = PAGE_CARDS.get(rel)
        if key is None:
            new_text, was_removed = _remove_block(text)
            was_inserted = False
            removed += int(was_removed)
        else:
            card = cards[key]
            block = _block(page, card)
            new_text, was_inserted = _insert_or_replace(text, block)
        if new_text != text:
            page.write_text(new_text, encoding="utf-8")
            changed += 1
            inserted += int(was_inserted)
    print(json.dumps({"pages": len(pages), "changed": changed, "inserted": inserted, "removed": removed}, indent=2))


if __name__ == "__main__":
    main()
