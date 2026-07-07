"""Tests for RemoteFileGateway's fetch-error diagnosis.

Regression coverage for a real, user-reported case: OpenNeuro's GraphQL
manifest returning a CDN URL/versionId that doesn't exist in S3 (confirmed by
direct investigation to be an upstream OpenNeuro data-integrity issue on a
specific dataset, not a Qortex caching bug — a fresh, uncached manifest fetch
returned the identical broken URL). ``_describe_fetch_error`` turns the raw
S3 XML error body into an honest, actionable message instead of dumping XML.
"""

from __future__ import annotations

from qortex.client.remote import _describe_fetch_error

# The exact S3 error body from the real user report (ds007932).
_REAL_NO_SUCH_VERSION_BODY = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<Error><Code>NoSuchVersion</Code>"
    "<Message>The specified version does not exist.</Message>"
    "<Key>ds007932/sub-105/func/sub-105_task-motor_acq-shimSlice 1mm sms2_run-02_bold.nii.gz</Key>"
    "<VersionId>gfwWjx3DEfW1.R3hKnnHRwvv3MK3VRLp</VersionId></Error>"
)

_NO_SUCH_KEY_BODY = (
    '<?xml version="1.0" encoding="UTF-8"?>\n'
    "<Error><Code>NoSuchKey</Code><Message>The specified key does not exist.</Message>"
    "<Key>ds000001/sub-01/anat/sub-01_T1w.nii.gz</Key></Error>"
)


class TestDescribeFetchError:
    def test_no_such_version_is_diagnosed_as_upstream_issue(self):
        msg = _describe_fetch_error(404, _REAL_NO_SUCH_VERSION_BODY)
        assert "upstream" in msg.lower()
        assert "not a" in msg.lower()  # explicitly rules out network/caching
        assert "NoSuchVersion" in msg

    def test_no_such_key_is_diagnosed_as_upstream_issue(self):
        msg = _describe_fetch_error(404, _NO_SUCH_KEY_BODY)
        assert "upstream" in msg.lower()
        assert "NoSuchKey" in msg

    def test_generic_error_falls_back_to_status_and_body(self):
        msg = _describe_fetch_error(500, "Internal Server Error")
        assert msg.startswith("HTTP 500:")
        assert "Internal Server Error" in msg

    def test_empty_body_does_not_crash(self):
        msg = _describe_fetch_error(403, "")
        assert msg.startswith("HTTP 403:")

    def test_unrelated_s3_error_code_falls_back_to_raw_body(self):
        # A real S3 error, but not one that means "doesn't exist" — should
        # not be misdiagnosed as the upstream-data-integrity case.
        body = "<Error><Code>AccessDenied</Code><Message>Access Denied</Message></Error>"
        msg = _describe_fetch_error(403, body)
        assert "upstream" not in msg.lower()
        assert msg.startswith("HTTP 403:")
