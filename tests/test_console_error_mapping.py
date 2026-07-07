"""The console API funnels every Qortex library call through `call()`, whose
job is to translate library exceptions into honest HTTP status codes for the
Atlas pages. A wrong mapping here is what makes an Events tab on a dataset
with no events, or a Preview of a path not in the snapshot, show a scary
"502 upstream failure" instead of a clean 404 — so it is worth pinning down.
"""

from __future__ import annotations

import asyncio

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi import HTTPException  # noqa: E402

from qortex.console.api import call  # noqa: E402
from qortex.core.exceptions import (  # noqa: E402
    DatasetNotFoundError,
    QortexError,
    RateLimitError,
)


def _status_for(exc: Exception) -> int:
    def _raise():
        raise exc

    try:
        asyncio.run(call(_raise))
    except HTTPException as http_exc:
        return http_exc.status_code
    raise AssertionError("call() did not raise")


def test_file_not_found_maps_to_404():
    # e.g. Dataset.events()/preview()/sidecar() raising for a missing path or
    # a dataset with no events file — a real not-found, not a gateway failure.
    assert _status_for(FileNotFoundError("no events file in manifest")) == 404


def test_value_error_maps_to_400():
    assert _status_for(ValueError("axis out of range")) == 400


def test_key_error_maps_to_400():
    assert _status_for(KeyError("missing entity")) == 400


def test_qortex_error_maps_to_400():
    assert _status_for(QortexError("bad request")) == 400


def test_dataset_not_found_maps_to_404():
    assert _status_for(DatasetNotFoundError("ds999999")) == 404


def test_rate_limit_maps_to_429():
    assert _status_for(RateLimitError("slow down")) == 429


def test_unexpected_error_still_maps_to_502():
    # A genuinely unexpected failure (e.g. an upstream/transport error not
    # modelled above) must remain a 502 — the not-found/bad-input mappings
    # must not swallow real upstream problems.
    assert _status_for(RuntimeError("connection reset by peer")) == 502
