"""One fresh check of the sole authorized write filesystem per finite batch."""
from .fresh_quota import _run, parse_quota


def fresh_write_quota(*, run=_run):
    # Appsdata payloads and frozen catalogs are read-only in this workflow.
    # A failed/low scratch query aborts; no retry, polling or stale fallback.
    output=run(['lfs','quota','-g','bs0247','-q','/scratch'])
    return {'/scratch':parse_quota(output,'/scratch')}
