# Snapshots

A snapshot is a versioned, immutable copy of an OpenNeuro dataset. Every time a dataset depositor updates their data, a new snapshot is created with an incremented version number. Old snapshots remain accessible indefinitely.

## Why snapshots matter for reproducibility

If you pin a snapshot version, the manifest and all CDN file URLs remain constant. The same Qortex command will produce the same download regardless of when it runs.

If you do not pin a snapshot, Qortex uses the latest published snapshot at the time you first fetch the manifest. Subsequent runs may resolve to a different (newer) snapshot.

## List available snapshots

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    snapshots = client.get_snapshots("ds004130")

for s in snapshots:
    print(s.tag, s.created, s.size)
```

From the CLI:

```bash
qortex metadata ds004130 --snapshots
```

Output:

```
ds004130 snapshots:
  1.0.0   2022-03-15   87 subjects   4.1 GB
  1.1.0   2022-08-20   88 subjects   4.2 GB
  1.2.0   2023-01-10   88 subjects   4.2 GB  ← latest
```

## Pin a snapshot

```python
from qortex import Dataset

ds = Dataset("ds004130", snapshot="1.0.0")
profile = ds.inspect()
print(profile.n_subjects)  # 87
```

From the CLI, add `--snapshot`:

```bash
qortex inspect ds004130 --snapshot 1.0.0
qortex download ds004130 --snapshot 1.0.0 --output-dir data/ds004130_v1/
```

## Get summary for a specific snapshot

```python
from qortex.client import OpenNeuroClient

with OpenNeuroClient() as client:
    summary = client.get_snapshot_summary("ds004130", "1.0.0")

print(summary.n_subjects)    # 87
print(summary.total_size_gb) # 4.1
print(summary.bids_version)  # "1.6.0"
print(summary.tasks)         # ["rest"]
print(summary.sessions)      # []
print(summary.funding)       # ["NIH ..."]
```

## Compare snapshots

To see which files changed between two snapshots:

```python
from qortex import Dataset

ds_old = Dataset("ds004130", snapshot="1.0.0")
ds_new = Dataset("ds004130", snapshot="1.2.0")

old_paths = {f.path for f in ds_old.manifest().files}
new_paths = {f.path for f in ds_new.manifest().files}

added   = new_paths - old_paths
removed = old_paths - new_paths
print(f"Added: {len(added)}  Removed: {len(removed)}")
```

## Recording the snapshot in provenance

When you convert a dataset, the snapshot version is recorded in the artifact manifest automatically:

```python
art = ds.convert(...)
print(art.manifest.source_snapshot)  # "ds004130@1.2.0"
```

This lets you trace any artifact back to the exact dataset version it came from.

## Limitations

- Very old snapshots (pre-2020) may have stale CDN URLs that return 404. Use a newer snapshot if possible.
- The snapshot list is fetched from the OpenNeuro GraphQL API. A network error here means you cannot list snapshots.








<!-- qortex-evidence:start -->

## Evidence

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/ds000001-manifest-composition.png" alt="Bar charts showing OpenNeuro ds000001 file suffix counts and bytes by BIDS datatype.">
  <figcaption>Real `Dataset.manifest()` output from OpenNeuro ds000001: suffix counts and bytes by BIDS datatype.</figcaption>
</figure>

```python
ds = Dataset('ds000001', snapshot='1.0.0')
manifest = ds.manifest()
```

Result artifact: [ds000001-example-results.json](/Qortex/assets/results/ds000001-example-results.json)

<!-- qortex-evidence:end -->
