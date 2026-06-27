"""
Fetch live statistics from the OpenNeuro GraphQL API and write them to
docs/assets/data/openneuro-stats.json.

Run locally:
    python scripts/update_openneuro_stats.py

Run in CI: see .github/workflows/openneuro-stats.yml
"""

import json
import sys
from pathlib import Path

import requests

ENDPOINT = "https://openneuro.org/crn/graphql"
PAGE_SIZE = 100

QUERY = """
query OpenNeuroStats($first: Int, $after: String) {
  datasets(first: $first, after: $after) {
    edges {
      node {
        id
        metadata {
          modalities
        }
        latestSnapshot {
          summary {
            subjects
            modalities
          }
        }
      }
    }
    pageInfo {
      hasNextPage
      endCursor
    }
  }
}
"""

CANONICAL_MODALITY_MAP = {
    "mri": "MRI",
    "fmri": "MRI",
    "bold": "MRI",
    "t1w": "MRI",
    "t2w": "MRI",
    "flair": "MRI",
    "dwi": "MRI",
    "dmri": "MRI",
    "dti": "MRI",
    "anat": "MRI",
    "func": "MRI",
    "perf": "MRI",
    "pet": "PET",
    "meg": "MEG",
    "eeg": "EEG",
    "ieeg": "iEEG",
    "eeg/ieeg": "iEEG",
    "ecog": "iEEG",
    "seeg": "iEEG",
    "nirs": "NIRS",
    "fnirs": "NIRS",
    "beh": "BEH",
    "behavioral": "BEH",
}

CANONICAL_ORDER = ["MRI", "PET", "MEG", "EEG", "iEEG", "NIRS", "BEH"]


def canonicalize(raw: str) -> str | None:
    return CANONICAL_MODALITY_MAP.get(raw.strip().lower())


def fetch_page(after: str | None) -> dict | None:
    """Return a page of datasets or None when the API has no more results.

    GraphQL may return both `data` and `errors` in the same response when
    individual dataset records are unavailable (deleted, access-restricted,
    or returning 'Not Found').  We treat that as a partial page — we use the
    data that IS present and continue paginating.  We only stop when:
      - the response has errors but NO data at all, or
      - pageInfo.hasNextPage is False.
    """
    response = requests.post(
        ENDPOINT,
        json={"query": QUERY, "variables": {"first": PAGE_SIZE, "after": after}},
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()

    # Check for data first — GraphQL may return partial data alongside errors.
    data = payload.get("data") or {}
    page = data.get("datasets")

    if payload.get("errors"):
        messages = [e.get("message", "") for e in payload["errors"]]
        if page is None:
            # Fatal: no data at all.  Treat as end-of-results.
            print(
                f"  GraphQL fatal errors (stopping): {'; '.join(messages[:3])}",
                file=sys.stderr,
            )
            return None
        # Partial errors alongside valid data — log and continue.
        print(
            f"  Partial errors (continuing): {'; '.join(messages[:3])}",
            file=sys.stderr,
        )

    return page


def collect_stats() -> dict:
    after = None
    public_datasets = 0
    participants = 0
    modalities: set[str] = set()
    page_num = 0
    consecutive_empty = 0

    while True:
        page_num += 1
        print(f"  page {page_num} (after={after!r})…", file=sys.stderr)
        page = fetch_page(after)

        if page is None:
            print("  No data returned — stopping.", file=sys.stderr)
            break

        edges = page.get("edges") or []

        if not edges:
            consecutive_empty += 1
            if consecutive_empty >= 3:
                print("  Three empty pages in a row — stopping.", file=sys.stderr)
                break
        else:
            consecutive_empty = 0

        for edge in edges:
            if not edge:
                continue
            node = edge.get("node") or {}
            if not node:
                continue
            public_datasets += 1

            summary = (node.get("latestSnapshot") or {}).get("summary") or {}
            subjects = summary.get("subjects") or []
            participants += len(subjects)

            raw_modalities = (node.get("metadata") or {}).get("modalities") or []
            raw_modalities = list(raw_modalities) + list(summary.get("modalities") or [])

            for raw in raw_modalities:
                if raw:
                    canonical = canonicalize(str(raw))
                    if canonical:
                        modalities.add(canonical)

        page_info = page.get("pageInfo") or {}
        if not page_info.get("hasNextPage"):
            print(f"  hasNextPage=False — done.", file=sys.stderr)
            break

        next_cursor = page_info.get("endCursor")
        if not next_cursor:
            print("  No endCursor — stopping.", file=sys.stderr)
            break
        after = next_cursor

    canonical_found = [m for m in CANONICAL_ORDER if m in modalities]
    extra = sorted(modalities - set(CANONICAL_ORDER))

    return {
        "public_datasets": public_datasets,
        "participants": participants,
        "modalities": canonical_found + extra,
    }


def main() -> None:
    out_path = Path("docs/assets/data/openneuro-stats.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print("Fetching OpenNeuro stats…", file=sys.stderr)
    stats = collect_stats()

    out_path.write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(json.dumps(stats, indent=2))
    print(f"\nWritten to {out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
