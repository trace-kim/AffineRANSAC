"""What the machine is doing, for the detailed tile-loading log (docs/SPEC.md D149): when this process was created,
how much of the machine's CPU is in use, and its memory. Windows only (MaskReg's platform), read from the Windows API
without a new dependency; elsewhere these report nothing (None, "")."""

import ctypes
import os

WINDOWS = os.name == "nt"
EPOCH_S = 11644473600  # Windows counts time from 1601-01-01, time.time() from 1970-01-01
GB = 2 ** 30

if WINDOWS:
    from ctypes import wintypes

    _kernel32 = ctypes.WinDLL("kernel32")  # its own function objects: setting their types here changes no one else's
    _kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    _kernel32.GetProcessTimes.argtypes = [wintypes.HANDLE] + [ctypes.POINTER(wintypes.FILETIME)] * 4

    class _MemoryStatus(ctypes.Structure):  # MEMORYSTATUSEX
        _fields_ = [("length", wintypes.DWORD), ("load", wintypes.DWORD)] + [
            (name, ctypes.c_uint64) for name in ("total_ram", "free_ram", "commit_limit", "commit_free",
                                                 "total_virtual", "free_virtual", "free_extended")]


def _seconds(filetime) -> float:
    """A Windows FILETIME (counts of 100 ns) in seconds."""
    return (filetime.dwHighDateTime << 32 | filetime.dwLowDateTime) / 1e7


def process_created() -> float | None:
    """When this process was created, in time.time() seconds; None where unknown."""
    if not WINDOWS:
        return None
    created, exited, kernel, user = (wintypes.FILETIME() for _ in range(4))
    if not _kernel32.GetProcessTimes(_kernel32.GetCurrentProcess(), created, exited, kernel, user):
        return None
    return _seconds(created) - EPOCH_S


def _cpu_times() -> tuple[float, float] | None:
    """(idle, busy + idle) CPU seconds of the whole machine since it started, summed over its CPUs."""
    if not WINDOWS:
        return None
    idle, kernel, user = (wintypes.FILETIME() for _ in range(3))
    if not _kernel32.GetSystemTimes(ctypes.byref(idle), ctypes.byref(kernel), ctypes.byref(user)):
        return None
    return _seconds(idle), _seconds(kernel) + _seconds(user)  # kernel time includes the idle time


def memory_gb() -> tuple[float, float, float, float] | None:
    """(RAM in use, RAM installed, memory committed, commit limit) in GB. Committed: what all programs have been
    promised; it can grow up to the limit (RAM + page file)."""
    if not WINDOWS:
        return None
    status = _MemoryStatus(length=ctypes.sizeof(_MemoryStatus))
    if not _kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        return None
    return ((status.total_ram - status.free_ram) / GB, status.total_ram / GB,
            (status.commit_limit - status.commit_free) / GB, status.commit_limit / GB)


class MachineLoad:
    """text(): the machine's CPU in use since the previous text() (or since this was made), and its memory now."""

    def __init__(self):
        self.last = _cpu_times()

    def cpu_percent(self) -> float | None:
        now, last = _cpu_times(), self.last
        self.last = now
        if now is None or last is None or now[1] <= last[1]:
            return None
        return 100 * (1 - (now[0] - last[0]) / (now[1] - last[1]))

    def text(self) -> str:
        """E.g. "CPU 3 %, RAM in use 41 of 384 GB, committed 60 of 420 GB"; "" where unknown."""
        cpu, memory = self.cpu_percent(), memory_gb()
        parts = [] if cpu is None else [f"CPU {cpu:.0f} %"]
        if memory is not None:
            parts.append(f"RAM in use {memory[0]:.0f} of {memory[1]:.0f} GB, committed {memory[2]:.0f} of "
                         f"{memory[3]:.0f} GB")
        return ", ".join(parts)
