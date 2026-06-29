# Dataset API

`Dataset` is the main entry point for working with an OpenNeuro or local BIDS dataset. Creating an instance makes no network calls — methods that need the manifest fetch it on first call.

```python
from qortex import Dataset

ds = Dataset("ds004130")
ds = Dataset("ds004130", snapshot="1.2.0")
ds = Dataset("ds004130", data_dir="/data/ds004130")
```

::: qortex.Dataset
    options:
      show_source: false
      members:
        - manifest
        - info
        - files
        - metadata_files
        - inspect
        - participants
        - events
        - sidecar
        - nifti_info
        - preview
        - first_rows
        - preview_metadata
        - prefetch_metadata
        - stream_header
        - stream_slice
        - get_lazy_array
        - map_labels
        - label_landscape
        - signal_budget
        - doctor
        - minimum
        - can_train
        - first_batch
        - content_status
        - check
        - plan
        - select
        - download
        - download_metadata
        - download_paths
        - validate
        - index_local
        - eda
        - convert
        - train_test_split
        - with_format
        - torch_dataset
        - to_torch_dataloader
        - lightning_datamodule
        - sklearn_arrays
        - to_monai_dicts
        - to_torcheeg_epochs
        - visualize
        - visual_audit
