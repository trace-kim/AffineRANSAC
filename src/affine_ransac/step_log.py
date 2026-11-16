"""Where the time of a search goes: seconds per step and counts (docs/SPEC.md D157; the contour diagnosis log)."""

import time
from contextlib import contextmanager


class StepLog:
    """The seconds each named step took, in the order the steps first ran, and counts of what they worked on."""

    def __init__(self):
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add(self, step: str, seconds: float):
        """Add to a step's time (a step run several times adds up)."""
        self.seconds[step] = self.seconds.get(step, 0.0) + seconds


@contextmanager
def timed(log: StepLog | None, step: str):
    """Time the block as this step of the log; no log: nothing."""
    start = time.perf_counter()
    try:
        yield
    finally:
        if log is not None:
            log.add(step, time.perf_counter() - start)
