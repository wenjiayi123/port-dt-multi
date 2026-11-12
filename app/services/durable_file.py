"""Publish a flushed sibling file durably on either supported host."""
import os


def publish(temporary, target):
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        move = ctypes.WinDLL("kernel32", use_last_error=True).MoveFileExW
        move.argtypes = (wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD)
        move.restype = wintypes.BOOL
        if not move(str(temporary), str(target), 0x1 | 0x8):  # replace, write-through
            raise ctypes.WinError(ctypes.get_last_error())
        return
    temporary.replace(target)
    descriptor = os.open(target.parent, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
