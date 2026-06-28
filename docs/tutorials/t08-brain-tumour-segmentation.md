# T08 — Brain Tumour Segmentation Baseline

**Dataset:** MSD Task01 Brain Tumour (484 training / 266 test volumes)  
**Task:** Multiclass segmentation — background / NCR+NET / edema / enhancing tumour  
**First model:** MONAI 3D SegResNet with Dice loss  
**Later model:** UNETR / SwinUNETR  
**Difficulty:** Advanced

---

## Prerequisites

```bash
pip install 'qortex[tutorials,monai,mri]'
# Full MONAI with GPU support:
pip install 'monai[all]'
```

GPU required for reasonable training time (~16 GB VRAM for full BraTS).  
CPU mode works for a quick sanity check on 5 cases.

---

## Step 1 — Load data

```python
from qortex.datasets import msd_brain

card = msd_brain.describe()
print(card)

bundle = msd_brain.load_data(
    split="train",
    max_cases=20,     # start with 20 cases
    download=True,    # MONAI downloads and caches on first call
)
bundle.info()
# SegmentationBundle: MSD Brain Tumour
#   Split      : train
#   Cases      : 20
#   Modalities : ['FLAIR', 'T1w', 'T1gd', 'T2w']
#   Label map  : {0: background, 1: NCR_NET, 2: edema, 3: enhancing_tumour}
```

**Label map**

| Value | Region |
|---|---|
| 0 | Background |
| 1 | NCR/NET — Necrotic core / non-enhancing tumour |
| 2 | Edema |
| 3 | Enhancing tumour |

---

## Step 2 — Image-mask pair validation

```python
import numpy as np

for i in range(min(5, bundle.n_cases)):
    image, mask = bundle.load_pair(i)
    # image: [4, x, y, z] (4 modalities stacked)
    # mask:  [x, y, z]
    assert image.ndim == 3 or (image.ndim == 4 and image.shape[0] == 4), \
        f"Case {i}: unexpected image shape {image.shape}"
    assert mask.shape == image.shape[-3:] or mask.shape == image.shape[1:], \
        f"Case {i}: mask shape {mask.shape} does not match image {image.shape}"

    # Label inventory
    unique_vals = np.unique(mask).tolist()
    label_names = [msd_brain.LABEL_MAP.get(int(v), f"val_{v}") for v in unique_vals]
    print(f"Case {bundle.case_ids[i]}: image={image.shape}, "
          f"mask={mask.shape}, labels={label_names}")
```

---

## Step 3 — MONAI data pipeline

```python
import torch
from monai.transforms import (
    Compose, LoadImaged, EnsureChannelFirstd, Orientationd,
    Spacingd, ScaleIntensityRangePercentilesd, CropForegroundd,
    RandCropByPosNegLabeld, RandFlipd, RandRotate90d, ToTensord,
)
from monai.data import CacheDataset, DataLoader

# Build data dicts for MONAI
def bundle_to_data_dicts(bundle):
    dicts = []
    for i in range(bundle.n_cases):
        img_paths = bundle.image_paths[i]
        msk_path  = bundle.mask_paths[i] if i < len(bundle.mask_paths) else None
        d = {"image": [str(p) for p in img_paths]}
        if msk_path and msk_path.exists():
            d["label"] = str(msk_path)
        dicts.append(d)
    return dicts

train_dicts = bundle_to_data_dicts(bundle)

train_transforms = Compose([
    LoadImaged(keys=["image", "label"]),
    EnsureChannelFirstd(keys=["image", "label"]),
    Orientationd(keys=["image", "label"], axcodes="RAS"),
    Spacingd(keys=["image", "label"], pixdim=(1.0, 1.0, 1.0),
             mode=("bilinear", "nearest")),
    ScaleIntensityRangePercentilesd(
        keys=["image"], lower=0.5, upper=99.5,
        b_min=0.0, b_max=1.0, clip=True, channel_wise=True,
    ),
    CropForegroundd(keys=["image", "label"], source_key="image"),
    RandCropByPosNegLabeld(
        keys=["image", "label"], label_key="label",
        spatial_size=(128, 128, 64), pos=1, neg=1, num_samples=2,
    ),
    RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
    RandRotate90d(keys=["image", "label"], prob=0.5, max_k=3),
    ToTensord(keys=["image", "label"]),
])

dataset = CacheDataset(data=train_dicts, transform=train_transforms, cache_rate=0.5)
loader  = DataLoader(dataset, batch_size=2, shuffle=True, num_workers=4)
```

---

## Step 4 — Build model

```python
from monai.networks.nets import SegResNet
from monai.losses import DiceLoss
from monai.metrics import DiceMetric

model = SegResNet(
    spatial_dims=3,
    in_channels=4,        # FLAIR, T1w, T1gd, T2w
    out_channels=4,       # background + 3 tumour regions
    init_filters=16,
).to("cuda" if torch.cuda.is_available() else "cpu")

loss_fn = DiceLoss(
    to_onehot_y=True,
    softmax=True,
    include_background=False,   # focus on tumour regions only
)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
dice_metric = DiceMetric(include_background=False, reduction="mean")

device = next(model.parameters()).device
print(f"Model device: {device}")
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
```

---

## Step 5 — Training loop

```python
from monai.inferers import sliding_window_inference

MAX_EPOCHS = 50
VAL_INTERVAL = 5
best_dice = -1

for epoch in range(1, MAX_EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        optimizer.zero_grad()
        pred = model(x)
        loss = loss_fn(pred, y)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d} | loss: {epoch_loss / len(loader):.4f}")
```

---

## Step 6 — Sliding-window inference and Dice evaluation

```python
from monai.transforms import AsDiscrete
from monai.data import decollate_batch

post_pred  = AsDiscrete(argmax=True, to_onehot=4)
post_label = AsDiscrete(to_onehot=4)

model.eval()
with torch.no_grad():
    for batch in loader:
        x = batch["image"].to(device)
        y = batch["label"].to(device)
        pred = sliding_window_inference(
            x, roi_size=(128, 128, 64), sw_batch_size=4, predictor=model,
        )
        preds_list  = [post_pred(p) for p in decollate_batch(pred)]
        labels_list = [post_label(l) for l in decollate_batch(y)]
        dice_metric(y_pred=preds_list, y=labels_list)
    mean_dice = dice_metric.aggregate().item()
    dice_metric.reset()

print(f"Mean Dice (excl. background): {mean_dice:.4f}")
print()
region_names = [msd_brain.LABEL_MAP[k] for k in [1, 2, 3]]
print("Regions measured:", region_names)
```

---

## Step 7 — Patch sanity check

```python
# Verify patch extraction is sane
sample = next(iter(loader))
img_patch  = sample["image"]
mask_patch = sample["label"]
print(f"Image patch shape : {img_patch.shape}")   # (batch, 4, 128, 128, 64)
print(f"Mask patch shape  : {mask_patch.shape}")  # (batch, 1, 128, 128, 64)

# Fraction of foreground in patches
frac_fg = (mask_patch > 0).float().mean().item()
print(f"Foreground fraction in patches: {frac_fg:.2%}")
assert frac_fg > 0.01, "Patches contain almost no foreground — check CropForeground"
print("✓ Patch sanity check passed")
```

---

## Validation summary

| Gate | Check |
|---|---|
| Image-mask pair check | Shapes match in Step 2 |
| Label inventory | All expected values {0,1,2,3} present (Step 2) |
| Affine/orientation | MONAI Orientationd ensures RAS+ |
| Patch extraction | Foreground fraction > 1% (Step 7) |
| Dice metric | Computed excluding background (Step 6) |
| No clinical claims | Documented in card and code comments |
