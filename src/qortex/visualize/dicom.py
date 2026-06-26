"""DICOM series browser and inspection (panel 7 in the reference image).

Supports:
- Listing all series in a study directory (multi-series folder)
- Inspecting a single series: geometry, modality, orientation, window settings
- Generating an HTML study/series browser (table view)
- Loading a DICOM series as a sorted 3D numpy array (via SimpleITK or pydicom)

No PACS/DICOMweb in this module — that belongs to qortex.connectors.dicomweb.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


# ── DicomSeries ───────────────────────────────────────────────────────────────

class DicomSeries:
    """Metadata record for one DICOM series."""

    def __init__(
        self,
        series_uid: str,
        series_number: int,
        description: str,
        modality: str,
        n_images: int,
        date: str,
        time: str,
        rows: int,
        cols: int,
        pixel_spacing: tuple[float, float] | None,
        slice_thickness: float | None,
        window_center: float | None,
        window_width: float | None,
        manufacturer: str,
        files: list[Path],
    ) -> None:
        self.series_uid = series_uid
        self.series_number = series_number
        self.description = description
        self.modality = modality
        self.n_images = n_images
        self.date = date
        self.time = time
        self.rows = rows
        self.cols = cols
        self.pixel_spacing = pixel_spacing
        self.slice_thickness = slice_thickness
        self.window_center = window_center
        self.window_width = window_width
        self.manufacturer = manufacturer
        self.files = files

    @property
    def shape(self) -> tuple[int, int, int]:
        return (self.rows, self.cols, self.n_images)

    @property
    def spacing_str(self) -> str:
        if self.pixel_spacing and self.slice_thickness:
            return f"{self.pixel_spacing[0]:.2f}×{self.pixel_spacing[1]:.2f}×{self.slice_thickness:.2f} mm"
        return "unknown"

    def to_dict(self) -> dict:
        return {
            "series_number": self.series_number,
            "description": self.description,
            "modality": self.modality,
            "n_images": self.n_images,
            "shape": list(self.shape),
            "spacing": self.spacing_str,
            "date": self.date,
            "series_uid": self.series_uid,
        }


# ── Scanning helpers ──────────────────────────────────────────────────────────

def _dcm_files(directory: Path) -> list[Path]:
    files = []
    for f in directory.iterdir():
        if f.is_file() and (f.suffix.lower() in {".dcm", ".dicom", ".ima"} or not f.suffix):
            files.append(f)
    return sorted(files)


def _require_pydicom():
    try:
        import pydicom
        return pydicom
    except ImportError:
        raise ImportError("DICOM inspection requires pydicom: pip install pydicom")


def list_dicom_series(directory: Path | str) -> list[DicomSeries]:
    """Scan a directory and group DICOM files by SeriesInstanceUID.

    Returns one DicomSeries per unique series, sorted by SeriesNumber.

    Uses only header reads (stop_before_pixels=True) for speed.
    """
    pydicom = _require_pydicom()
    directory = Path(directory)
    files = _dcm_files(directory)
    if not files:
        # Try one level deep (study with subdirectories per series)
        for subdir in directory.iterdir():
            if subdir.is_dir():
                files.extend(_dcm_files(subdir))

    if not files:
        raise FileNotFoundError(f"No DICOM files found in {directory}")

    # Group by SeriesInstanceUID
    series_map: dict[str, list[Path]] = {}
    meta_map: dict[str, Any] = {}

    for f in files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=True, force=True)
            uid = str(getattr(ds, "SeriesInstanceUID", f"unknown_{len(series_map)}"))
            series_map.setdefault(uid, []).append(f)
            if uid not in meta_map:
                meta_map[uid] = ds
        except Exception as exc:
            log.debug("Skip %s: %s", f, exc)

    result = []
    for uid, file_list in series_map.items():
        ds = meta_map[uid]
        ps = getattr(ds, "PixelSpacing", None)
        st = getattr(ds, "SliceThickness", None)
        wc = getattr(ds, "WindowCenter", None)
        ww = getattr(ds, "WindowWidth", None)

        # WindowCenter/Width may be a list (multiple presets)
        if isinstance(wc, (list, tuple)):
            wc = wc[0]
        if isinstance(ww, (list, tuple)):
            ww = ww[0]

        result.append(DicomSeries(
            series_uid=uid,
            series_number=int(getattr(ds, "SeriesNumber", 999)),
            description=str(getattr(ds, "SeriesDescription", "")),
            modality=str(getattr(ds, "Modality", "MR")),
            n_images=len(file_list),
            date=str(getattr(ds, "SeriesDate", "")),
            time=str(getattr(ds, "SeriesTime", "")),
            rows=int(getattr(ds, "Rows", 0)),
            cols=int(getattr(ds, "Columns", 0)),
            pixel_spacing=(float(ps[0]), float(ps[1])) if ps else None,
            slice_thickness=float(st) if st else None,
            window_center=float(wc) if wc else None,
            window_width=float(ww) if ww else None,
            manufacturer=str(getattr(ds, "Manufacturer", "")),
            files=sorted(file_list),
        ))

    return sorted(result, key=lambda s: s.series_number)


def load_dicom_series(
    series: DicomSeries | Path | str,
    *,
    apply_rescale: bool = True,
    sort_by_position: bool = True,
) -> tuple[np.ndarray, dict]:
    """Load a DICOM series into a sorted 3D numpy array.

    Uses SimpleITK for reliable slice sorting when available; falls back to
    pydicom with position-based sorting.

    Returns
    -------
    (volume, metadata_dict)
    volume: float32 array shaped (rows, cols, n_slices)
    """
    if not isinstance(series, DicomSeries):
        series_dir = Path(series)
        series_list = list_dicom_series(series_dir)
        if not series_list:
            raise FileNotFoundError(f"No DICOM series found in {series_dir}")
        series = series_list[0]

    # Try SimpleITK first (reliable slice ordering)
    try:
        import SimpleITK as sitk
        reader = sitk.ImageSeriesReader()
        file_names = sitk.ImageSeriesReader.GetGDCMSeriesFileNames(
            str(series.files[0].parent),
            series.series_uid,
        )
        reader.SetFileNames(file_names)
        img = reader.Execute()
        arr = sitk.GetArrayFromImage(img).astype(np.float32)
        # SimpleITK returns (z, y, x) — we want (x, y, z) = (rows, cols, slices)
        arr = arr.transpose(2, 1, 0)
        meta = {
            "n_slices": arr.shape[2],
            "shape": arr.shape,
            "spacing": img.GetSpacing(),
            "origin": img.GetOrigin(),
            "backend": "SimpleITK",
        }
        return arr, meta
    except ImportError:
        log.debug("SimpleITK not available; using pydicom for series loading")
    except Exception as exc:
        log.debug("SimpleITK failed: %s; falling back to pydicom", exc)

    # Fallback: pydicom + position sorting
    pydicom = _require_pydicom()

    slices_data = []
    for f in series.files:
        try:
            ds = pydicom.dcmread(str(f), stop_before_pixels=False, force=True)
            arr = ds.pixel_array.astype(np.float32)
            if apply_rescale:
                slope = float(getattr(ds, "RescaleSlope", 1.0) or 1.0)
                intercept = float(getattr(ds, "RescaleIntercept", 0.0) or 0.0)
                arr = arr * slope + intercept
            pos = getattr(ds, "ImagePositionPatient", None)
            z_pos = float(pos[2]) if pos else float(getattr(ds, "InstanceNumber", len(slices_data)))
            slices_data.append((z_pos, arr, ds))
        except Exception as exc:
            log.debug("Skip %s: %s", f, exc)

    if not slices_data:
        raise RuntimeError("No readable DICOM slices in series")

    if sort_by_position:
        slices_data.sort(key=lambda x: x[0])

    volume = np.stack([s[1] for s in slices_data], axis=-1)  # (rows, cols, n_slices)
    first_ds = slices_data[0][2]
    ps = getattr(first_ds, "PixelSpacing", [1.0, 1.0])
    st = float(getattr(first_ds, "SliceThickness", 1.0) or 1.0)
    meta = {
        "n_slices": len(slices_data),
        "shape": volume.shape,
        "spacing": (float(ps[0]), float(ps[1]), st),
        "backend": "pydicom",
    }
    return volume, meta


# ── DicomSeriesBrowser ────────────────────────────────────────────────────────

class DicomSeriesBrowser:
    """HTML series browser for a DICOM study directory (panel 7 style).

    Lists all series with modality, image count, description, date,
    and a thumbnail preview of the middle slice.
    """

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)
        self._series: list[DicomSeries] | None = None
        self._study_meta: dict = {}

    def scan(self) -> list[DicomSeries]:
        if self._series is None:
            self._series = list_dicom_series(self.directory)
            if self._series:
                self._extract_study_meta()
        return self._series

    def _extract_study_meta(self) -> None:
        pydicom = _require_pydicom()
        first_series = self._series[0]
        if first_series.files:
            try:
                ds = pydicom.dcmread(str(first_series.files[0]), stop_before_pixels=True)
                self._study_meta = {
                    "study_description": str(getattr(ds, "StudyDescription", "")),
                    "patient_id": str(getattr(ds, "PatientID", "ANON")),
                    "patient_dob": str(getattr(ds, "PatientBirthDate", "")),
                    "patient_sex": str(getattr(ds, "PatientSex", "")),
                    "study_date": str(getattr(ds, "StudyDate", "")),
                    "institution": str(getattr(ds, "InstitutionName", "")),
                    "manufacturer": str(getattr(ds, "Manufacturer", "")),
                }
            except Exception:
                pass

    def _modality_icon(self, modality: str) -> str:
        icons = {"MR": "🧠", "CT": "🦴", "PT": "☢️", "US": "🔊", "XA": "📡", "CR": "📷"}
        return icons.get(modality.upper(), "📄")

    def to_html(self) -> str:
        try:
            series_list = self.scan()
        except Exception as exc:
            return self._error_html(str(exc))

        study = self._study_meta
        study_desc = study.get("study_description", str(self.directory.name))
        patient_id = study.get("patient_id", "ANON")
        dob = study.get("patient_dob", "")
        sex = study.get("patient_sex", "")
        study_date = study.get("study_date", "")

        # Format date
        if len(study_date) == 8:
            study_date = f"{study_date[:4]}-{study_date[4:6]}-{study_date[6:]}"
        if len(dob) == 8:
            dob = f"{dob[:4]}-{dob[4:6]}-{dob[6:]}"

        rows_html = ""
        for i, s in enumerate(series_list):
            icon = self._modality_icon(s.modality)
            selected = ' style="background:#1a2a3a;border-left:3px solid #6af"' if i == 0 else ""
            date_str = f"{s.date[:4]}-{s.date[4:6]}-{s.date[6:]}" if len(s.date) == 8 else s.date
            time_str = f"{s.time[:2]}:{s.time[2:4]}" if len(s.time) >= 4 else s.time
            rows_html += f"""
<tr{selected} onclick="selectSeries({i})" style="cursor:pointer">
  <td style="padding:8px 4px;text-align:center;font-size:1.2em">{icon}</td>
  <td style="padding:8px 12px;color:#ccc">{s.series_number}</td>
  <td style="padding:8px 12px;color:#fff;font-weight:500">{s.description or '—'}</td>
  <td style="padding:8px 12px;color:#6af">{s.modality}</td>
  <td style="padding:8px 12px;color:#aaa;text-align:right">{s.n_images}</td>
  <td style="padding:8px 12px;color:#888;font-size:0.85em">{date_str} {time_str}</td>
</tr>"""

        detail_items = ""
        for i, s in enumerate(series_list):
            display = "block" if i == 0 else "none"
            wc_str = f"WC {s.window_center:.0f} / WW {s.window_width:.0f}" if s.window_center else "auto"
            detail_items += f"""
<div id="detail_{i}" style="display:{display};padding:12px 0">
  <table style="border-collapse:collapse;font-size:0.85em">
    <tr><td style="color:#888;padding:3px 16px 3px 0">Description</td><td style="color:#ccc">{s.description}</td></tr>
    <tr><td style="color:#888;padding:3px 16px 3px 0">Modality</td><td style="color:#6af">{s.modality}</td></tr>
    <tr><td style="color:#888;padding:3px 16px 3px 0">Shape</td><td style="color:#ccc">{s.rows} × {s.cols} × {s.n_images}</td></tr>
    <tr><td style="color:#888;padding:3px 16px 3px 0">Spacing</td><td style="color:#ccc">{s.spacing_str}</td></tr>
    <tr><td style="color:#888;padding:3px 16px 3px 0">Window</td><td style="color:#ccc">{wc_str}</td></tr>
    <tr><td style="color:#888;padding:3px 16px 3px 0">Manufacturer</td><td style="color:#888">{s.manufacturer}</td></tr>
    <tr><td style="color:#888;padding:3px 16px 3px 0">Series UID</td><td style="color:#555;font-size:0.75em;word-break:break-all">{s.series_uid[:40]}…</td></tr>
  </table>
</div>"""

        js = f"""
function selectSeries(idx) {{
  var n = {len(series_list)};
  for (var i = 0; i < n; i++) {{
    document.getElementById('detail_'+i).style.display = (i===idx) ? 'block' : 'none';
  }}
  var rows = document.querySelectorAll('tr[onclick]');
  rows.forEach(function(r, i) {{
    r.style.background = (i===idx) ? '#1a2a3a' : '';
    r.style.borderLeft = (i===idx) ? '3px solid #6af' : '';
  }});
}}"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>DICOM Study Browser</title>
<style>
  body {{ background:#111; color:#ccc; font-family:sans-serif; margin:20px; }}
  h2 {{ color:#6af; margin-bottom:4px; font-size:1.3em; }}
  .study-header {{ margin-bottom:16px; font-size:0.85em; color:#888; }}
  .study-header span {{ color:#ccc; margin-right:24px; }}
  .layout {{ display:flex; gap:20px; }}
  .series-table {{ flex:1.5; }}
  .detail-panel {{ flex:1; background:#1a1a1a; border-radius:6px; padding:12px; min-width:260px; }}
  table {{ border-collapse:collapse; width:100%; }}
  thead th {{ color:#6af; font-weight:500; text-align:left; padding:6px 12px;
              border-bottom:1px solid #333; font-size:0.85em; }}
  tbody tr:hover {{ background:#1e2a36; }}
  .tag {{ display:inline-block; background:#1a3a5a; color:#6af; border-radius:3px;
          padding:1px 6px; font-size:0.75em; margin-left:6px; }}
</style>
</head>
<body>
<h2>DICOM Study Browser <span class="tag">Study</span></h2>
<div class="study-header">
  <span>Study: <b>{study_desc}</b></span>
  <span>Patient ID: <b>{patient_id}</b></span>
  <span>DOB: {dob}</span>
  <span>Sex: {sex}</span>
  <span>Date: {study_date}</span>
</div>
<div class="layout">
  <div class="series-table">
    <table>
      <thead>
        <tr>
          <th style="width:32px"></th>
          <th>#</th>
          <th>Description</th>
          <th>Modality</th>
          <th style="text-align:right">Images</th>
          <th>Date / Time</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
  <div class="detail-panel">
    <div style="color:#6af;font-size:0.85em;font-weight:600;margin-bottom:8px">Series Detail</div>
    {detail_items}
  </div>
</div>
<script>{js}</script>
</body>
</html>"""

    def _error_html(self, message: str) -> str:
        return f"""<!DOCTYPE html>
<html><head><meta charset="UTF-8"></head>
<body style="background:#111;color:#f96;font-family:monospace;margin:20px">
<h2>DICOM Series Browser</h2>
<p>Error scanning {self.directory}:</p>
<pre>{message}</pre>
</body></html>"""

    def __repr__(self) -> str:
        n = len(self._series) if self._series else "?"
        return f"DicomSeriesBrowser({self.directory}, {n} series)"
