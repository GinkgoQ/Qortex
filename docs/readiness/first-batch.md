# First Batch

`ds.first_batch()` tells you whether Qortex can show a first usable batch now. If local data or an artifact is missing, it returns the smallest plan needed to produce one.

## Python

```python
from qortex import Dataset

ds = Dataset("ds004130")
report = ds.first_batch(local_path="data/ds004130", target="trial_type")
print(report.to_text())
```

The returned report has:

- status and source
- row or sample summary when data is available
- a required download plan when data is missing
- suggested next command

## What it does internally

1. Checks an artifact if `artifact_path` is provided.
2. Checks local BIDS files if `local_path` is provided.
3. Falls back to `minimum(goal="first-batch")` when local data is missing.
4. Reports the concrete blocker and next action.

If any step fails, a descriptive exception is raised with the stage that failed.

## CLI

```bash
qortex first-batch --dataset ds004130 --local-path data/ds004130 --target trial_type
qortex first-batch --artifact artifacts/ds004130
```

Typical output when local data is missing:

```
Status : uncertain
Source : ds004130
Rows   : 0
Message: No local data path was provided; returning the smallest required first-batch plan.
```

## Debugging first_batch failures

Common failures and what they mean:

**`EventsNotFound`** — events.tsv missing for one of the selected subjects. Usually means the manifest is not consistent — some subjects have events and some do not.

**`EmptyWindowError`** — the window duration is longer than the shortest trial. Set a shorter `window_s`.

**`LFSPointerError`** — the downloaded file is a Git LFS pointer, not real data. The dataset uses DataLad or Git LFS and was not properly fetched. Try `git-annex get` on the raw dataset.

**`NoLabelError`** — the target column is not in events.tsv. Check the actual column names with `ds.events(subject="01")`.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-minimum-plan.png" alt="Horizontal bar chart of the ds000001 first-batch download plan and file sizes.">
  <figcaption>Real `minimum(goal='first-batch')` plan: metadata, sidecar, events, and one BOLD run.</figcaption>
</figure>

```python
plan = ds.minimum(goal='first-batch', output_dir=Path('data/ds000001'))
print(plan.to_text())
```

Result artifact: [ds000001-minimum-first-batch.txt](/Qortex/assets/results/ds000001-minimum-first-batch.txt)

<!-- qortex-evidence:end -->

## Related

- [Minimum](minimum.md) — understand the subject selection logic
- [Can train](can-train.md) — structured label-readiness report before running first_batch
- [Conversion pipeline](../conversion/pipeline.md) — what happens during the convert step
