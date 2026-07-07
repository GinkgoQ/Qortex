# External Segmentation Runners

Qortex can call file-based segmentation engines and then render their outputs with the same artifact tools used by NeuroAI pipelines.

This layer is deliberately small. Qortex validates paths, builds the command, runs the external process, captures stdout/stderr/timing, checks that the expected output exists, writes a provenance JSON file, and returns typed paths. It does not replace the model’s own preprocessing, weights, license, or inference code.

## Supported Boundaries

| Engine | Qortex role | Upstream project |
|---|---|---|
| MONAI bundles | Inspect contracts through NeuroAI adapters; render produced masks | [MONAI Model Zoo](https://github.com/Project-MONAI/model-zoo) |
| nnU-Net v2 | Run `nnUNetv2_predict` when installed | [MIC-DKFZ nnU-Net](https://github.com/MIC-DKFZ/nnUNet) |
| TotalSegmentator | Run `TotalSegmentator` when installed | [TotalSegmentator](https://github.com/wasserth/TotalSegmentator) |
| MedSAM / MedSAM2 | Store and render masks produced by a local prompt pipeline | [MedSAM](https://github.com/bowang-lab/MedSAM), [MedSAM2](https://github.com/bowang-lab/MedSAM2) |

## Python API

```python
from qortex.neuroai import (
    ExternalSegmentationRequest,
    run_external_segmentation,
    render_segmentation_showcase_from_files,
)

run = run_external_segmentation(
    ExternalSegmentationRequest(
        engine="totalsegmentator",
        image_path="case_001_ct.nii.gz",
        output_path="artifacts/case_001_total.nii.gz",
        task="total",
        device="gpu",
        timeout_s=1800,
        extra_args=("--ml",),
    )
)

figures = render_segmentation_showcase_from_files(
    image_path=run.image_path,
    prediction_mask_path=run.output_path,
    output_dir="artifacts/case_001_showcase",
    case_id="case_001",
    model_id="TotalSegmentator:total",
)
```

## nnU-Net

`nnUNetv2_predict` expects trained results to be available through the nnU-Net environment/configuration. If `model_folder` is provided, Qortex sets `nnUNet_results` for the subprocess. The command still uses nnU-Net’s own flags for dataset id, configuration, trainer, plans, and folds.

```python
from qortex.neuroai import ExternalSegmentationRequest, run_external_segmentation

result = run_external_segmentation(
    ExternalSegmentationRequest(
        engine="nnunet",
        image_path="nnunet_input/case_001_0000.nii.gz",
        output_path="nnunet_predictions",
        model_folder="/models/nnUNet_results",
        dataset_id=501,
        configuration="3d_fullres",
        trainer="nnUNetTrainer",
        plans="nnUNetPlans",
        folds=(0, 1, 2, 3, 4),
        device="cuda",
    )
)
```

## CLI Rendering

After any external model writes a mask, render inspection artifacts:

```bash
qortex neuroai render-segmentation-showcase case_001_t1w.nii.gz case_001_mask.nii.gz artifacts/case_001_showcase \
  --case-id case_001 \
  --model-id nnUNetv2:Dataset501 \
  --truth-mask case_001_truth.nii.gz \
  --class-labels-json '{"0":"background","1":"tumour"}'
```

The output directory contains PNG figures plus `metrics.json` and `showcase-manifest.json`.

<figure class="tq-figure">
  <img src="/Qortex/assets/images/examples/neuroai-showcase/segmentation-board.png" alt="Qortex segmentation showcase board with source slice, foreground candidate mask, overlay, contour, metrics, and class legend.">
  <figcaption>Renderer output from the documentation generation run. The local fixture uses a source-derived foreground candidate mask, not a trained model prediction.</figcaption>
</figure>
