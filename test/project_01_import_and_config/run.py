"""project_01_import_and_config

Verifies that the package is installed and that configuration works correctly:
env-var overrides, runtime reconfiguration, and Dataset facade construction.
"""

from __future__ import annotations

import os
import sys
import tempfile
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import banner, print_kv, require, require_equal, passed  # noqa: E402

import qortex
from qortex.core import (
    ConfigurationError,
    FileRecord,
    Manifest,
    ModelAdapterError,
    QortexConfig,
    QortexError,
    QortexWarning,
    SplitPlan,
    emit_warning,
)


def main() -> None:
    banner("project_01: package import and runtime configuration")

    # ── 1. version present ────────────────────────────────────────────────────
    require(isinstance(qortex.__version__, str) and qortex.__version__, "version string missing")
    print_kv("version", qortex.__version__)

    # ── 2. configure() and get_config() ──────────────────────────────────────
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "qortex_cache"
        qortex.configure(
            cache_dir=cache_dir,
            max_concurrent_downloads=4,
            metadata_timeout=12.0,
        )
        cfg = qortex.get_config()

        print_kv("config", {
            "cache_dir": cfg.cache_dir,
            "max_concurrent_downloads": cfg.max_concurrent_downloads,
            "metadata_timeout": cfg.metadata_timeout,
        })

        require(cfg.cache_dir == cache_dir.resolve(), "cache_dir not applied")
        require_equal(cfg.max_concurrent_downloads, 4, "max_concurrent_downloads")
        require_equal(cfg.metadata_timeout, 12.0, "metadata_timeout")

    # ── 3. QortexConfig env-var prefix ───────────────────────────────────────
    os.environ["QORTEX_MAX_CONCURRENT_DOWNLOADS"] = "7"
    env_cfg = QortexConfig()
    require_equal(env_cfg.max_concurrent_downloads, 7, "env-var QORTEX_MAX_CONCURRENT_DOWNLOADS")
    del os.environ["QORTEX_MAX_CONCURRENT_DOWNLOADS"]

    # ── 4. Dataset facade construction ────────────────────────────────────────
    ds = qortex.Dataset("ds000001")
    require_equal(ds.dataset_id, "ds000001", "Dataset.dataset_id")
    require(callable(ds.manifest), "Dataset.manifest not callable")
    require(callable(ds.inspect), "Dataset.inspect not callable")
    require(callable(ds.participants), "Dataset.participants not callable")
    require(callable(ds.events), "Dataset.events not callable")
    require(callable(ds.sidecar), "Dataset.sidecar not callable")
    require(callable(ds.nifti_info), "Dataset.nifti_info not callable")
    require(callable(ds.label_landscape), "Dataset.label_landscape not callable")
    require(callable(ds.signal_budget), "Dataset.signal_budget not callable")

    # ── 5. Public API surface ─────────────────────────────────────────────────
    require(callable(qortex.search), "qortex.search missing")
    require(callable(qortex.configure), "qortex.configure missing")
    require(qortex.FileRecord is not None, "qortex.FileRecord missing")
    require(qortex.Manifest is not None, "qortex.Manifest missing")

    # ── 6. Core config safety ────────────────────────────────────────────────
    try:
        qortex.configure(max_concurrent_downloads=0)
    except ConfigurationError as exc:
        print_kv("invalid config error", exc.to_dict())
        require_equal(exc.code, "config.error", "ConfigurationError.code")
    else:
        raise RuntimeError("invalid config override did not raise ConfigurationError")

    # ── 7. Structured exceptions and warnings ────────────────────────────────
    err = QortexError(
        "example failure",
        code="example.failure",
        context={"stage": "project_01"},
        suggestion="Inspect the structured context.",
    )
    print_kv("structured error", err.to_dict())
    require_equal(err.to_dict()["code"], "example.failure", "QortexError.to_dict code")
    require(ModelAdapterError("blocked", model_id="m", provider="plugin").context["model_id"] == "m",
            "ModelAdapterError context missing")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        record = emit_warning(
            "core.test_warning",
            "structured warning example",
            context={"stage": "project_01"},
        )
    require_equal(record.code, "core.test_warning", "WarningRecord.code")
    require(caught and issubclass(caught[0].category, QortexWarning), "QortexWarning was not emitted")

    # ── 8. Core entity invariants ────────────────────────────────────────────
    try:
        FileRecord(id="bad", path="bad.txt", filename="bad.txt", extension=".txt", size=-1)
    except ValueError:
        pass
    else:
        raise RuntimeError("FileRecord accepted negative size")

    good_file = FileRecord(id="ok", path="sub-01/file.txt", filename="file.txt", extension=".txt", size=1)
    try:
        Manifest(dataset_id="ds", snapshot="1", files=[good_file, good_file])
    except ValueError:
        pass
    else:
        raise RuntimeError("Manifest accepted duplicate file paths")

    try:
        SplitPlan(train=0.8, val=0.2, test=0.2)
    except ValueError:
        pass
    else:
        raise RuntimeError("SplitPlan accepted fractions that do not sum to 1.0")

    passed("project_01_import_and_config")


if __name__ == "__main__":
    main()
