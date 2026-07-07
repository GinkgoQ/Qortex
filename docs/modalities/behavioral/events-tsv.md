# Events TSV

`events.tsv` is the BIDS standard for task timing and labels. It is the primary source of supervised learning labels for signal recordings (EEG, MEG, fMRI, iEEG, fNIRS).

## Structure

```
onset    duration  trial_type    response_time  stim_file
0.000    1.500     fixation      n/a            n/a
1.500    1.500     go            0.412          stimuli/go_001.png
3.000    1.500     nogo          n/a            stimuli/nogo_001.png
4.500    1.500     go            0.398          stimuli/go_002.png
```

**Required columns:**

| Column | Description |
|--------|-------------|
| `onset` | Event start time relative to recording start (seconds) |
| `duration` | Event duration in seconds (0 for instantaneous) |

**Recommended:**

| Column | Description |
|--------|-------------|
| `trial_type` | Primary label for ML; categorical condition name |
| `response_time` | Time from stimulus to response (seconds) — for regression tasks |
| `stim_file` | Path to stimulus file relative to dataset root |
| `HED` | Hierarchical Event Descriptors (for annotation-based workflows) |

## Naming convention

Each `events.tsv` is linked to one recording via its filename entities:

```
sub-01/func/sub-01_task-nback_events.tsv      → sub-01_task-nback_bold.nii.gz
sub-01/eeg/sub-01_task-rest_run-01_events.tsv → sub-01_task-rest_run-01_eeg.set
```

## Read events.tsv

```python
from qortex import Dataset

ds = Dataset("ds004130")

# From CDN (before download — events.tsv are small)
events = ds.events(subject="01", task="rest")

# After metadata download
events = ds.events(subject="01", task="rest")  # reads from local file

print(events.dtypes)        # column types
print(events.shape)         # (n_trials, n_columns)
print(events.head(5))
```

## Missing events.tsv

Qortex reports missing events.tsv in the doctor check:

```python
report = ds.doctor()
# ERROR [NO_EVENTS]: no events.tsv found for any subject
# WARNING [PARTIAL_EVENTS]: sub-03, sub-17 missing events.tsv
```

Datasets with no events.tsv can still be downloaded and used for unsupervised or regression tasks where labels come from `participants.tsv`.

## events.json sidecar

An optional `events.json` describes columns in `events.tsv`:

```json
{
  "trial_type": {
    "LongName": "Trial type",
    "Description": "Experimental condition",
    "Levels": {
      "go": "Go trial — respond to stimulus",
      "nogo": "No-go trial — withhold response"
    }
  }
}
```

Qortex does not currently parse `events.json` for column descriptions, but it is available via:

```python
desc = ds.sidecar("sub-01/func/sub-01_task-flanker_events.json")
print(desc["trial_type"]["Levels"])
```








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

- [Labels and trial types](labels-and-trial-types.md)
- [Label readiness](../../readiness/label-readiness.md)
- [Event-aligned windows](../../conversion/windows.md)
