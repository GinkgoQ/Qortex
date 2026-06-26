# OpenNeuro Troubleshooting

## GraphQL API errors

**`APIError: Failed to fetch manifest`**

The OpenNeuro GraphQL API is unreachable. Check:

1. Network connectivity: `curl https://openneuro.org/crn/graphql`
2. OpenNeuro status: https://status.openneuro.org

If the API is up and you still get errors, the dataset ID may not exist:

```bash
qortex inspect ds999999  # non-existent dataset → clear error message
```

**`APIError: Dataset not found`**

The dataset ID does not exist in OpenNeuro. Check the ID at https://openneuro.org/datasets/{dataset_id}.

**`APIError: Unauthorized`**

The dataset is private or embargoed. Store an API token:

```bash
qortex login
# Prompts for token from https://openneuro.org/profile/access-tokens
```

## Rate limiting

OpenNeuro's GraphQL endpoint has rate limits. If you are making many manifest requests in a loop, spread them out:

```python
import time
from qortex import Dataset

datasets = ["ds000001", "ds000002", ...]
reports = []
for ds_id in datasets:
    ds = Dataset(ds_id)
    reports.append(ds.doctor())
    time.sleep(1)  # 1 second between requests
```

## Snapshot not found

```
SnapshotNotFoundError: Snapshot 1.0.0 not found for ds004130
```

Use `qortex metadata ds004130 --snapshots` to list available snapshot versions.

## Manifest caching issues

If you get stale manifest data, force a refresh:

```python
ds = Dataset("ds004130")
manifest = ds.manifest(force_refresh=True)
```

Or delete the cached manifest:

```bash
rm ~/.qortex/manifests/ds004130/1.2.0.json
```
