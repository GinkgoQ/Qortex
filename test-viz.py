# from pathlib import Path

# from qortex import Dataset
# import qortex.visualize as qv


# DATASET_ID = "ds000001"
# OUT_DIR = Path("data/qortex_visual_test")

# # Important: current DownloadPlanner expects target parent to already exist
# OUT_DIR.mkdir(parents=True, exist_ok=True)

# ds = Dataset(DATASET_ID)
# manifest = ds.manifest()

# visual_candidates = [
#     f for f in manifest.files
#     if f.path.endswith((".nii", ".nii.gz", ".mgz", ".mgh"))
#     and (f.size is None or f.size < 300_000_000)
# ]

# if not visual_candidates:
#     raise RuntimeError("No visualizable NIfTI/MGH file found.")

# priority = {
#     "T1w": 0,
#     "T2w": 1,
#     "FLAIR": 2,
#     "bold": 3,
#     "dwi": 4,
# }

# visual_candidates.sort(
#     key=lambda f: (
#         priority.get(f.suffix or "", 99),
#         f.size or 0,
#     )
# )

# file_record = visual_candidates[0]

# print("Selected OpenNeuro file:")
# print(" ", file_record.path)
# print(" suffix:", file_record.suffix)
# print(" size MB:", round((file_record.size or 0) / 1e6, 2))

# result = ds.download_paths(
#     [file_record.path],
#     output_dir=OUT_DIR,
#     with_companions=True,
#     max_size_gb=0.5,
# )

# print(result.report())

# local_file = OUT_DIR / file_record.path

# if not local_file.exists():
#     raise FileNotFoundError(
#         f"Expected downloaded file not found: {local_file}")

# asset = qv.inspect(local_file)
# print(asset.summary())
# print(asset.plan().describe())

# # Load the data before plotting
# viewer = qv.volume(local_file)

# print("Loaded data:")
# print(" shape:", viewer.shape)
# print(" voxel sizes:", viewer.voxel_sizes)
# print(" volumes:", viewer.n_volumes)

# fig = viewer.ortho(title=f"Qortex Visualize — {file_record.path}")
# fig.show()

# viewer.to_html(
#     OUT_DIR / "qortex_visual_viewer.html",
#     title=f"Qortex Visualize — {file_record.path}",
# )

from pathlib import Path
from qortex import Dataset

ds = Dataset("ds000001")
m = ds.manifest()
p = ds.plan(subjects=["01"], metadata_only=True)
r = ds.download_metadata(output_dir=Path("tmp/ds000001-meta"))
doctor = ds.doctor(local_path=Path("tmp/ds000001-meta"))
eda = ds.eda(local_path=Path("tmp/ds000001-meta"))
print(eda)
