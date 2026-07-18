from pathlib import Path

from qortex.console.cache_inventory import cache_inventory
from qortex.core.config import QortexConfig
from qortex.neuroai.models.cache import ModelCache


def test_inventory_measures_real_files_and_does_not_follow_symlinks(tmp_path: Path) -> None:
    core = tmp_path / "core"
    home = tmp_path / "home"
    (core / "catalog").mkdir(parents=True)
    (core / "catalog" / "catalog.duckdb").write_bytes(b"catalog")
    external = tmp_path / "external.bin"
    external.write_bytes(b"not-counted")
    (core / "catalog" / "external-link").symlink_to(external)
    stream = home / ".qortex" / "stream_cache"
    stream.mkdir(parents=True)
    (stream / "range.bin").write_bytes(b"range")

    result = cache_inventory(
        QortexConfig(cache_dir=core),
        home=home,
        model_cache=ModelCache(home / ".qortex" / "model_cache"),
    )
    by_id = {item["id"]: item for item in result["surfaces"]}

    assert by_id["catalog"]["file_count"] == 1
    assert by_id["catalog"]["size_bytes"] == len(b"catalog")
    assert by_id["stream"]["size_bytes"] == len(b"range")
    assert by_id["stream"]["max_bytes"] == 2_000_000_000
    assert by_id["stream"]["ttl_seconds"] == 3600
    assert result["total_bytes"] == len(b"catalog") + len(b"range")


def test_inventory_reports_absent_surfaces_without_creating_them(tmp_path: Path) -> None:
    core = tmp_path / "core"
    home = tmp_path / "home"
    result = cache_inventory(
        QortexConfig(cache_dir=core),
        home=home,
        model_cache=ModelCache(home / ".qortex" / "model_cache"),
    )

    assert all(item["exists"] is False for item in result["surfaces"])
    assert not core.exists()
    assert not home.exists()
