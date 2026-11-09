"""Reusable lfs command processes; every request executes a fresh quota command.

This does not cache quota data. Callers must retain their writer lock and error
latch. A malformed, failed or timed-out response permanently poisons a session.
"""
import os
import selectors
import subprocess
import threading
import time

FOOTER = (b'usage: lfs [COMMAND] [OPTIONS]... [ARGS]\n'
          b'Without any parameters, interactive mode is invoked\n'
          b"Try 'lfs help <COMMAND>' or 'lfs --list-commands' for more information\n")


class QuotaSession:
    def __init__(self, filesystem, *, command=None, timeout=45):
        if filesystem not in ('/scratch', '/appsdata'):
            raise ValueError('Unexpected filesystem')
        self.filesystem = filesystem
        self.timeout = timeout
        self.lock = threading.Lock()
        self.failed = False
        self.process = subprocess.Popen(command or ['stdbuf', '-oL', '-eL', 'lfs'],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            bufsize=0)
        self.selector = selectors.DefaultSelector()
        self.selector.register(self.process.stdout, selectors.EVENT_READ)

    def read_fresh(self):
        with self.lock:
            if self.failed or self.process.poll() is not None:
                raise RuntimeError('Quota session unavailable')
            try:
                # No prior buffered response may satisfy a new command.
                if self.selector.select(0):
                    raise RuntimeError('Unexpected queued quota output')
                message = f'quota -g bs0247 -q {self.filesystem}\nhelp\n'.encode()
                self.process.stdin.write(message)
                output = bytearray()
                end = time.monotonic() + self.timeout
                while True:
                    remaining = end-time.monotonic()
                    if remaining <= 0 or not self.selector.select(remaining):
                        raise TimeoutError('Fresh quota command timed out')
                    chunk = os.read(self.process.stdout.fileno(), 65536)
                    if not chunk:
                        raise RuntimeError('Quota command process exited')
                    output.extend(chunk)
                    if len(output) > 65536:
                        raise RuntimeError('Unexpected quota output size')
                    if FOOTER in output:
                        if not output.endswith(FOOTER) or output.count(FOOTER) != 1:
                            raise RuntimeError('Ambiguous quota response boundary')
                        body = bytes(output[:-len(FOOTER)]).decode()
                        rows = [line for line in body.splitlines() if line.strip()]
                        if len(rows) != 1 or rows[0].split()[0] != self.filesystem:
                            raise RuntimeError('Unexpected quota diagnostic or filesystem')
                        # Caller applies strict numeric parser and reserve checks.
                        return rows[0]+'\n'
            except BaseException:
                self.failed = True
                self.close()
                raise

    def close(self):
        self.failed = True
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=2)
        self.selector.close()
        for stream in (self.process.stdin, self.process.stdout):
            if stream:
                stream.close()
