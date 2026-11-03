"""The machine's CPU and memory, and when a process was created (machine.py, D149): Windows only."""

import os
import re
import time

import pytest

from affine_ransac import machine

pytestmark = pytest.mark.skipif(os.name != "nt", reason="read from the Windows API")


def test_this_process_was_created_before_now_and_today():
    created = machine.process_created()
    assert created is not None and time.time() - 86400 < created < time.time()


def test_memory_in_use_is_within_what_there_is():
    in_use, installed, committed, limit = machine.memory_gb()
    assert 0 < in_use <= installed and 0 < committed <= limit and installed <= limit


def test_machine_load_names_cpu_and_memory():
    load = machine.MachineLoad()
    sum(range(10 ** 6))  # a little CPU time between the two readings
    percent = load.cpu_percent()
    assert percent is None or 0 <= percent <= 100
    assert re.fullmatch(r"(CPU \d+ %, )?RAM in use \d+ of \d+ GB, committed \d+ of \d+ GB", load.text())
