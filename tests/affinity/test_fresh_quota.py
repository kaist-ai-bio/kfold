import importlib.util
from pathlib import Path
import subprocess
import threading

import pytest

spec = importlib.util.spec_from_file_location(
    'fresh_quota', Path(__file__).parents[2] / 'src/kfold/affinity/fresh_quota.py')
quota = importlib.util.module_from_spec(spec)
spec.loader.exec_module(quota)


def row(fs, used=1_000_000_000, soft=10_000_000_000, hard=12_000_000_000,
        files=10, fsoft=5000000, fhard=6000000):
    return f'{fs} {used} {soft} {hard} - {files} {fsoft} {fhard} -\n'


def test_two_fresh_queries_per_call_and_mount_concurrency():
    calls = []
    barrier = threading.Barrier(2)
    def run(args):
        calls.append(args[-1])
        barrier.wait(timeout=5)
        return row(args[-1])
    for _ in range(2):
        result = quota.fresh_quota(run=run)
        assert result['/scratch']['free_bytes'] == 9_000_000_000 * 1024
    assert sorted(calls) == ['/appsdata', '/appsdata', '/scratch', '/scratch']


def test_one_mount_timeout_fails_whole_check():
    def run(args):
        if args[-1] == '/appsdata':
            raise subprocess.TimeoutExpired(args, 45)
        return row(args[-1])
    with pytest.raises(subprocess.TimeoutExpired):
        quota.fresh_quota(run=run)


@pytest.mark.parametrize('output', [
    '', '/scratch malformed', row('/scratch') * 2,
    row('/scratch', soft=0, hard=0), row('/scratch', files=-1),
])
def test_unknown_quota_fails_closed(output):
    with pytest.raises(ValueError):
        quota.parse_quota(output, '/scratch')


@pytest.mark.parametrize('output', [
    row('/scratch', used=9_000_000_000),
    row('/scratch', files=4_999_001),
])
def test_reserves_preserved(output):
    with pytest.raises(RuntimeError):
        quota.parse_quota(output, '/scratch')


def test_star_grace_and_tighter_limit():
    output = row('/scratch', soft=12_000_000_000, hard=10_000_000_000).replace('1000000000 ', '1000000000* ', 1)
    assert quota.parse_quota(output, '/scratch')['free_bytes'] == 9_000_000_000 * 1024
