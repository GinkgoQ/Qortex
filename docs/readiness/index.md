# Readiness

Readiness checks answer "is this dataset usable for training?" before you download anything expensive. Each check works on the manifest and small sidecar files — not on imaging data.

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Real ds000001 events.tsv timeline used to inspect label candidates">
  <figcaption>Real event evidence from `ds000001` subject `01`, run `01`. Qortex can see candidate labels remotely, but it reports uncertainty until local label columns and policies are confirmed.</figcaption>
</figure>

## Checks and what they answer

[**Doctor**](doctor.md) — Full structural readiness report. Covers subjects, modalities, companion files, label coverage, split feasibility, and total size. Start here.

[**Minimum**](minimum.md) — What is the smallest download that achieves a specific goal? Returns a file list with sizes.

[**Can train**](can-train.md) — Binary check: does this dataset have enough labeled samples to train a classifier? Takes a target column and minimum requirements.

[**First batch**](first-batch.md) — Return a minimal subset of subjects sufficient to run one complete pipeline pass end-to-end.

[**Label readiness**](label-readiness.md) — Per-subject label coverage, class counts, and missing subjects. Works against either manifest or local events.tsv files.

[**Content status**](content-status.md) — After download, verify that local files are complete and not Git LFS pointers.

[**Recipes**](recipes.md) — Predefined task-specific readiness recipes (e.g., "fmri-classification", "eeg-regression") that bundle the most common check parameters.

[**Leakage check**](leakage-check.md) — After conversion, verify that no subject appears in two splits.

## Reading readiness reports

All readiness methods return structured Pydantic objects with:

- `state` — a string enum (e.g., `"not_usable"`, `"manifest_only"`, `"download_ready"`)
- `findings` — list of `Finding` objects with severity, code, and message
- `next_action` — a string describing the recommended next step
- `to_text()` — human-readable summary
- `to_dict()` / `to_json()` — machine-readable output

```python
report = ds.doctor()
if report.state == "not_usable":
    print(report.to_text())
    for f in report.findings:
        if f.severity == "error":
            print(f"  ERROR: {f.message}")
```

---

## LabelPolicy — deterministic label enforcement

By default `compute_readiness()` scans events.tsv files for any of the seven standard BIDS label columns (`trial_type`, `stim_type`, `condition`, …). This heuristic is permissive: it reports `label_ready=True` as long as *any* matching column name is present, regardless of content.

For training you usually want to enforce a specific column and validate its values. Pass an explicit `LabelPolicy`:

```python
from qortex.core.entities import LabelPolicy
from qortex.check.readiness import compute_readiness

policy = LabelPolicy(
    source="events",
    column="trial_type",        # exact column name required
    task="rest",
    missing="drop",             # "drop" | "error" — what to do with null rows
    positive_values=["target", "probe"],  # expected label values
)

report = compute_readiness(manifest, local_path=bids_root, label_policy=policy)
```

When a `LabelPolicy` is provided, the readiness checker enforces:

| Check | Failure code | Severity |
|---|---|---|
| Column `policy.column` must exist | `labels.policy_column_missing` | warning |
| Column must not be all-null | `labels.policy_column_all_null` | warning |
| Null fraction + `missing="error"` | `labels.policy_missing_values` | error |
| At least one `positive_values` present | `labels.policy_no_positive_values` | warning |

Without `local_path` the checker cannot open events.tsv on disk and falls back to manifest-level presence detection, reporting `labels.candidate_unverified` so you can see that policy enforcement was skipped.

---

**Next →** [Download](../download/index.md) — once readiness is confirmed, fetch exactly the files you need.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-can-train.png" alt="Readiness chart for ds000001 showing subject count, recording count, and label-ready count.">
  <figcaption>Real `CanTrainReport` for ds000001. Qortex separates candidate labels from locally confirmed training evidence.</figcaption>
</figure>

```python
report = ds.can_train(target='trial_type')
print(report.to_text())
```

Result artifact: [ds000001-can-train.txt](/Qortex/assets/results/ds000001-can-train.txt)

<!-- qortex-evidence:end -->
