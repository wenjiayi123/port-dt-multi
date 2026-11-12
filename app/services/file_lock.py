"""Cross-process lock for the receipt and trial ledgers."""
from contextlib import contextmanager
import errno
import os
import time

if os.name == "nt":
    import msvcrt
else:
    import fcntl


@contextmanager
def exclusive_lock(path):
    with path.open("a+b") as stream:
        if os.name != "nt":
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)
            return
        # Lock the same byte even when the lock file is empty.
        stream.seek(0)
        while True:
            try:
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                break
            except OSError as error:
                if error.errno not in (errno.EACCES, errno.EDEADLK):
                    raise
                time.sleep(0.05)
        try:
            yield
        finally:
            stream.seek(0)
            msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
