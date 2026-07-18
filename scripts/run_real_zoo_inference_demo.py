"""Run the pinned pretrained BraTS bundle against a real public MRI case."""

from __future__ import annotations

import argparse
import json

from qortex.neuroai.public_validation import (
    DEFAULT_CASE_ID,
    run_public_brats_validation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Download pinned public MONAI/BraTS artifacts, run pretrained inference, "
            "and persist predictions, Dice metrics, checksums, and provenance."
        )
    )
    parser.add_argument("--case-id", default=DEFAULT_CASE_ID)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    result = run_public_brats_validation(case_id=args.case_id, device=args.device)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
