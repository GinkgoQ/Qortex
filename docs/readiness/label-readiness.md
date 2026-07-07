# Label Readiness

Label readiness tells you whether a dataset has enough labeled samples to train a classifier. It complements `can_train()`: the readiness report says whether to proceed, while `label_landscape()` explains the class structure that will shape the model.

## Python

```python
from qortex import Dataset

ds = Dataset("ds000001", snapshot="1.0.0")
landscape = ds.label_landscape(label_column="trial_type", max_events_files=4)
print(landscape.summary())
```

Output:

```
Label Landscape — ds000001
Events files: 4/4 fetched
Label column: trial_type
Classes: 4  Total events: 568
Coverage: 100.0% of signal keys have events
Imbalance: 7.87x (severe)
Cross-subject consistency: 100.0%

Class distribution:
  pumps_demean                       299  ███████████████
  control_pumps_demean               187  █████████
  cash_demean                         44  ██
  explode_demean                      38  ██
```

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Real ds000001 event timeline showing four trial type classes across the first run.">
  <figcaption>Real events from `Dataset("ds000001").events(subject="01", task="balloonanalogrisktask", run="01")`. The visible class imbalance is why `label_landscape()` belongs before model training.</figcaption>
</figure>

## What label_landscape shows

- Which events.tsv column was selected (or the column you specified)
- Per-class sample counts summed across all subjects
- Which subjects are missing each class
- Whether label coverage is balanced

## Specify the label column

```python
landscape = ds.label_landscape(label_column="response_type")
```

If `label_column` is `None`, Qortex auto-selects the column with the most distinct non-null values after excluding timing and provenance columns such as `onset`, `duration`, `stim_file`, `HED`, and `response_time`.

## CLI

```bash
qortex eda ds004130 --label trial_type
```

The `eda` command runs exploratory checks that include `label_landscape` plus signal statistics.

## Working from local events.tsv files

If you have already downloaded the metadata:

```python
ds = Dataset("ds004130", data_dir="data/ds004130/")
landscape = ds.label_landscape()  # reads from local events.tsv files
```

Without a local directory, `label_landscape()` fetches `events.tsv` files from CDN. This is fast because events files are small and Qortex does not fetch the paired imaging data.

## Interpreting class imbalance

A class with fewer samples per subject is not automatically a problem. If `task` appears only once per 10-minute run while `rest` appears every 20 seconds, the imbalance is by design. Review the events.tsv structure before deciding to exclude subjects or classes.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-events-timeline.png" alt="Timeline of ds000001 events and trial-type counts for subject 01 run 01.">
  <figcaption>Real `events.tsv` timeline for ds000001 sub-01 run-01.</figcaption>
</figure>

```python
events = ds.events(subject='01', task='balloonanalogrisktask', run='01')
print(events.shape)
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->

## Related

- [Can train](can-train.md) — structured training-readiness report
- [Doctor](doctor.md) — includes label landscape in the full report
- [EDA](../conversion/eda.md) — full exploratory data analysis including signal statistics
