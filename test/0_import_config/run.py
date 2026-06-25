from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import qortex

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from project_support import banner, print_kv, require  # noqa: E402


def main() -> None:
    banner("PROJECT 0: installed package and runtime configuration")
    with tempfile.TemporaryDirectory() as tmp:
        cache_dir = Path(tmp) / "cache"
        qortex.configure(
            cache_dir=cache_dir,
            max_concurrent_downloads=3,
            metadata_timeout=7.5,
        )
        cfg = qortex.get_config()
        ds = qortex.Dataset("ds000001")

        print_kv(
            "Observed output",
            {
                "qortex version": qortex.__version__,
                "configured cache": cfg.cache_dir,
                "download workers": cfg.max_concurrent_downloads,
                "metadata timeout": cfg.metadata_timeout,
                "dataset facade id": ds.dataset_id,
                "public search callable": callable(qortex.search),
            },
        )

        require(cfg.cache_dir == cache_dir.resolve(), "cache_dir override was not applied")
        require(cfg.max_concurrent_downloads == 3, "download concurrency override was not applied")
        require(ds.dataset_id == "ds000001", "Dataset facade did not preserve dataset_id")
        require(callable(qortex.search), "qortex.search is not available")

    print("RESULT: project 0 passed")


if __name__ == "__main__":
    main()
