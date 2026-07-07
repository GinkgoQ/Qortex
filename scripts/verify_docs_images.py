"""Verify that every local docs image reference points to a real file."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DOCS = ROOT / "docs"
SITE = ROOT / "site"


IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp")


def _references(text: str) -> list[str]:
    refs = re.findall(r'<img[^>]+src="([^"]+)"', text)
    refs.extend(re.findall(r"!\[[^\]]*\]\(([^)]+)\)", text))
    return refs


def _target_for_doc(page: Path, ref: str) -> Path | None:
    clean = ref.split("#", 1)[0].split("?", 1)[0].strip()
    if not clean or clean.startswith(("http://", "https://", "data:", "mailto:")):
        return None
    if not clean.lower().endswith(IMAGE_EXTENSIONS):
        return None
    if clean.startswith("/Qortex/"):
        return DOCS / clean.removeprefix("/Qortex/")
    if clean.startswith("/"):
        return DOCS / clean.lstrip("/")
    return (page.parent / clean).resolve()


def _target_for_site(page: Path, ref: str) -> Path | None:
    clean = ref.split("#", 1)[0].split("?", 1)[0].strip()
    if not clean or clean.startswith(("http://", "https://", "data:", "mailto:")):
        return None
    if not clean.lower().endswith(IMAGE_EXTENSIONS):
        return None
    if clean.startswith("/Qortex/"):
        return SITE / clean.removeprefix("/Qortex/")
    if clean.startswith("/"):
        return SITE / clean.lstrip("/")
    return (page.parent / clean).resolve()


def _check_tree(root: Path, target_fn) -> list[tuple[str, str, str]]:
    missing: list[tuple[str, str, str]] = []
    for page in sorted(root.rglob("*")):
        if page.suffix.lower() not in {".md", ".html"}:
            continue
        text = page.read_text(encoding="utf-8", errors="ignore")
        for ref in _references(text):
            target = target_fn(page, ref)
            if target is not None and not target.exists():
                missing.append((str(page.relative_to(ROOT)), ref, str(target)))
    return missing


def main() -> int:
    missing = _check_tree(DOCS, _target_for_doc)
    if SITE.exists():
        missing.extend(_check_tree(SITE, _target_for_site))
    if missing:
        for page, ref, target in missing:
            print(f"MISSING\t{page}\t{ref}\t{target}")
        return 1
    print("docs image references: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
