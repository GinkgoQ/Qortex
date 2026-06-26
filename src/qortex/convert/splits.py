"""Train / val / test split strategies.

All splitters operate on lists of SampleRecord and return three lists.
Subject-aware splits ensure no subject appears in more than one partition.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from typing import TypeAlias

from qortex.core.entities import SampleRecord

log = logging.getLogger(__name__)

SplitTriple: TypeAlias = tuple[
    list[SampleRecord], list[SampleRecord], list[SampleRecord]
]


@dataclass(frozen=True)
class SplitSpec:
    train: float = 0.7
    val: float = 0.15
    test: float = 0.15
    seed: int = 42
    stratify_by_label: bool = True
    strategy: str = "subject"  # "subject" | "random" | "stratified"

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if not (0.999 < total < 1.001):
            raise ValueError(f"Split fractions must sum to 1.0, got {total:.3f}")


def subject_split(samples: list[SampleRecord], spec: SplitSpec) -> SplitTriple:
    """Assign whole subjects to splits — no subject leaks between sets.

    Samples with subject=None are routed to train with a warning rather than
    silently dropped, which would produce mismatched artifact row counts.
    """
    no_subject = [s for s in samples if not s.subject]
    if no_subject:
        log.warning(
            "%d sample(s) have no subject ID and will be assigned to train. "
            "Inspect provenance to verify these are not primary data records.",
            len(no_subject),
        )
        for s in no_subject:
            s.split = "train"

    subjects = sorted({s.subject for s in samples if s.subject})
    rng = random.Random(spec.seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, int(round(spec.train * n))) if n > 0 else 0
    n_val = max(0, int(round(spec.val * n)))

    train_subs = set(subjects[:n_train])
    val_subs = set(subjects[n_train : n_train + n_val])
    test_subs = set(subjects[n_train + n_val :])

    def _mark(lst: list[SampleRecord], split: str) -> list[SampleRecord]:
        for s in lst:
            s.split = split
        return lst

    train = _mark([s for s in samples if s.subject in train_subs], "train") + no_subject
    val = _mark([s for s in samples if s.subject in val_subs], "val")
    test = _mark([s for s in samples if s.subject in test_subs], "test")
    return train, val, test


def random_split(samples: list[SampleRecord], spec: SplitSpec) -> SplitTriple:
    """Simple random shuffle — may leak subjects across splits."""
    items = list(samples)
    rng = random.Random(spec.seed)
    rng.shuffle(items)

    n = len(items)
    n_train = int(round(spec.train * n))
    n_val = int(round(spec.val * n))

    train = items[:n_train]
    val = items[n_train : n_train + n_val]
    test = items[n_train + n_val :]
    for s in train:
        s.split = "train"
    for s in val:
        s.split = "val"
    for s in test:
        s.split = "test"
    return train, val, test


def stratified_subject_split(
    samples: list[SampleRecord], spec: SplitSpec
) -> SplitTriple:
    """Stratified by label: preserve class balance across splits.

    Subjects are assigned to label buckets by their majority label, then split
    fractions are applied per bucket.  Falls back to subject_split when labels
    are unavailable.  Subjectless samples are routed to train (see subject_split).
    """
    if not any(s.label is not None for s in samples):
        return subject_split(samples, spec)

    no_subject = [s for s in samples if not s.subject]
    if no_subject:
        log.warning(
            "%d sample(s) have no subject ID; routing to train in stratified split.",
            len(no_subject),
        )
        for s in no_subject:
            s.split = "train"

    sub_label: dict[str, dict[int, int]] = {}
    for s in samples:
        if not s.subject or s.label is None:
            continue
        bucket = sub_label.setdefault(s.subject, {})
        bucket[s.label] = bucket.get(s.label, 0) + 1

    sub_majority: dict[str, int] = {
        sub: max(cnt, key=cnt.__getitem__) for sub, cnt in sub_label.items()
    }
    label_to_subs: dict[int, list[str]] = {}
    for sub, lbl in sub_majority.items():
        label_to_subs.setdefault(lbl, []).append(sub)

    rng = random.Random(spec.seed)
    train_subs: set[str] = set()
    val_subs: set[str] = set()
    test_subs: set[str] = set()

    for _lbl, subs in label_to_subs.items():
        subs_s = sorted(subs)
        rng.shuffle(subs_s)
        n = len(subs_s)
        n_train = max(1, int(round(spec.train * n)))
        n_val = max(0, int(round(spec.val * n)))
        train_subs.update(subs_s[:n_train])
        val_subs.update(subs_s[n_train : n_train + n_val])
        test_subs.update(subs_s[n_train + n_val :])

    def _mark(lst: list[SampleRecord], split: str) -> list[SampleRecord]:
        for s in lst:
            s.split = split
        return lst

    train = _mark([s for s in samples if s.subject in train_subs], "train") + no_subject
    val = _mark([s for s in samples if s.subject in val_subs], "val")
    test = _mark([s for s in samples if s.subject in test_subs], "test")
    return train, val, test


def apply_split(
    samples: list[SampleRecord], spec: SplitSpec
) -> SplitTriple:
    """Dispatch to the configured split strategy.

    When strategy="subject" and stratify_by_label=True, this promotes to
    stratified subject-level splitting so class balance is preserved across
    train/val/test partitions.
    """
    strategy = spec.strategy

    # Honour stratify_by_label on the default "subject" strategy: if the user
    # sets stratify_by_label=True without changing strategy, we promote to the
    # stratified path rather than silently ignoring the flag.
    if strategy == "subject" and spec.stratify_by_label:
        return stratified_subject_split(samples, spec)
    elif strategy == "subject":
        return subject_split(samples, spec)
    elif strategy == "random":
        return random_split(samples, spec)
    elif strategy == "stratified":
        return stratified_subject_split(samples, spec)
    else:
        raise ValueError(
            f"Unknown split strategy '{strategy}'. "
            f"Available: subject, random, stratified"
        )
