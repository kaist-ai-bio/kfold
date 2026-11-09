"""Fresh two-mount quotas with process reuse, never quota-value reuse."""
import atexit
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from multiprocessing.util import Finalize

from .fresh_quota import FILESYSTEMS, parse_quota
from .quota_session import QuotaSession


class SessionQuota:
    def __init__(self, factory=QuotaSession):
        self.sessions = {}
        self.owner = os.getpid()
        self.lock = threading.Lock()
        self.failed = False
        try:
            for fs in FILESYSTEMS:
                self.sessions[fs] = factory(fs)
        except BaseException:
            self.close()
            raise

    def __call__(self):
        with self.lock:
            if self.failed or self.owner != os.getpid():
                raise RuntimeError('Unavailable or inherited quota sessions')
            try:
                with ThreadPoolExecutor(max_workers=2) as pool:
                    futures = {fs: pool.submit(s.read_fresh) for fs,s in self.sessions.items()}
                    result = {fs: parse_quota(f.result(), fs) for fs,f in futures.items()}
                return result
            except BaseException:
                self.close()
                raise

    def close(self):
        self.failed = True
        for session in self.sessions.values():
            session.close()


_provider = None
_finalizer = None


def fresh_quota():
    global _provider, _finalizer
    if _provider is None:
        _provider = SessionQuota()
        _finalizer = Finalize(None, _provider.close, exitpriority=10)
        atexit.register(_provider.close)
    return _provider()
