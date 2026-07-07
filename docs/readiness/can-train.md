# Can Train

`ds.can_train()` answers one narrow question: does the dataset have enough confirmed label evidence to support supervised training for the requested target?

It returns a `CanTrainReport`, not a bare boolean. The report preserves uncertainty: event files can exist while labels remain unconfirmed until local columns are inspected.

## Real Example

```python
from qortex import Dataset

ds = Dataset("ds000001")
report = ds.can_train(target="trial_type")
print(report.to_text())
```

Observed output:

```text
Dataset  : ds000001 (1.0.0)
Status   : uncertain
Modality : any
Target   : trial_type
Labels   : candidate
Subjects : 16
Records  : 80
Ready    : 0
Required : 50.5 MB
Split    : subject
```

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-can-train.png" alt="CanTrainReport for ds000001 showing uncertain status, candidate labels, 16 subjects, 80 recordings, and zero label-ready recordings">
  <figcaption>Real `CanTrainReport` from `ds000001`. The manifest contains event files, but Qortex refuses to call the dataset training-ready until local label columns are verified or an explicit label policy is used.</figcaption>
</figure>

## Reading The Report

| Field | Meaning in the real example |
|---|---|
| `status` | `uncertain`: enough structure exists to continue, but label evidence is not confirmed. |
| `label_status` | `candidate`: matching `events.tsv` files exist. |
| `n_subjects` | 16 subjects in the manifest. |
| `n_recordings` | 80 recording records considered by the decision report. |
| `n_label_ready` | 0 because local label columns have not been confirmed. |
| `required_download_bytes` | 50.5 MB, the first-batch payload needed to move forward. |
| `suggested_split` | `subject`, because samples from the same participant must not cross splits. |

## Confirm Labels Locally

Download the smallest label-check or first-batch plan, then rerun `can_train()` with `local_path`:

```python
label_plan = ds.minimum(goal="label-check", output_dir="data/ds000001-labels")
ds.download_paths(label_plan.plan.files, output_dir="data/ds000001-labels")

confirmed = ds.can_train(
    target="trial_type",
    local_path="data/ds000001-labels",
)
print(confirmed.to_text())
```

For production conversion, use an explicit label policy when you know the correct column and accepted values.

## CLI

```bash
qortex can-train ds000001 --target trial_type
qortex minimum ds000001 --goal label-check --download --output-dir data/ds000001-labels
qortex can-train ds000001 --target trial_type --local-path data/ds000001-labels
```

## What Can Train Checks

| Check | Why it matters |
|---|---|
| Event files | Supervised labels usually come from `events.tsv`. |
| Target column | The requested target must exist locally before labels are confirmed. |
| Label-ready records | Training needs records whose label evidence is not merely inferred. |
| Subject count | Splits need enough independent participants. |
| Split policy | Subject-level splits prevent leakage from repeated measures. |
| Required download | The report tells you the smallest next payload needed to resolve uncertainty. |

## What It Does Not Prove

`can_train()` does not certify scientific validity, motion quality, class balance for every downstream metric, or clinical usefulness. It proves that Qortex has enough label and split evidence to proceed with a supervised training workflow.








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

## Related

- [Doctor](doctor.md)
- [Label readiness](label-readiness.md)
- [Minimum](minimum.md)
- [First batch](first-batch.md)
