"""Fresh quota checks for distinct filesystems; never reuse a snapshot."""
from concurrent.futures import ThreadPoolExecutor
import subprocess


FILESYSTEMS = ('/scratch', '/appsdata')


def _run(args):
    return subprocess.run(args, check=True, capture_output=True, text=True,
                          timeout=45).stdout


def parse_quota(output, filesystem, *, reserve_bytes=3_500_000_000_000,
                reserve_files=1000):
    rows = [line.split() for line in output.splitlines()
            if line.split() and line.split()[0] == filesystem]
    if len(rows) != 1 or len(rows[0]) != 9:
        raise ValueError('Unrecognized quota for ' + filesystem)
    row = rows[0]
    used, soft, hard, files, fsoft, fhard = (
        int(row[i].rstrip('*')) for i in (1, 2, 3, 5, 6, 7))
    if min(used, soft, hard, files, fsoft, fhard) < 0:
        raise ValueError('Negative quota field')
    byte_limits = [v for v in (soft, hard) if v > 0]
    file_limits = [v for v in (fsoft, fhard) if v > 0]
    if not byte_limits or not file_limits:
        raise ValueError('No finite quota limit')
    free_bytes = (min(byte_limits) - used) * 1024
    free_files = min(file_limits) - files
    if free_bytes < reserve_bytes or free_files < reserve_files:
        raise RuntimeError('Quota reserve reached: ' + filesystem)
    return dict(free_bytes=free_bytes, free_files=free_files)


def fresh_quota(*, run=_run):
    """Query both mounts anew, concurrently, within the caller's writer lock.

    At most one query per mount is in flight. Both must pass before returning.
    The caller retains its cross-process lock, error latch and write forecasts.
    No retries, stale fallback, shared snapshots or background work are used.
    """
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {fs: executor.submit(run, ['lfs', 'quota', '-g', 'bs0247', '-q', fs])
                   for fs in FILESYSTEMS}
        return {fs: parse_quota(futures[fs].result(), fs) for fs in FILESYSTEMS}
