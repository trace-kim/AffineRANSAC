"""D166: memory-based admission, bounded pending results and cleanup without changing task order."""

import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import closing
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from affine_ransac import contour_memory as memory


def headers(monkeypatch, sizes):
    monkeypatch.setattr(memory, "image_pixels", lambda path: np.prod(sizes[path]) if min(sizes[path]) > 0 else None)


def test_worker_admission_accounts_for_upscale_ram_commit_and_existing_pool(monkeypatch):
    paths = [str(k) for k in range(20)]
    headers(monkeypatch, dict.fromkeys(paths, (2048, 2048)))
    monkeypatch.setattr(memory, "worker_count", lambda: 8)
    monkeypatch.setattr(memory, "memory_gb", lambda: (20, 32, 30, 50))  # 12 GiB RAM free, 20 GiB commit free
    log = []
    assert memory.contour_workers(paths, 2) == 7
    assert memory.contour_workers(paths, 4, log.append) == 3
    assert memory.contour_workers(paths, 8) == 1
    assert "estimated allowance 2.25 GiB" in log[0]
    assert memory.contour_workers(paths, 4, pool=SimpleNamespace(_max_workers=2)) == 2
    assert memory.contour_workers(paths[:2], 4) == 2
    monkeypatch.setattr(memory, "memory_gb", lambda: (5, 32, 48, 50))  # commit, not RAM, limits allocations
    assert memory.contour_workers(paths, 4) == 1


def test_largest_image_is_used_and_unknown_header_or_memory_allows_one_task(monkeypatch):
    headers(monkeypatch, {"small": (256, 256), "large": (4096, 4096), "unknown": (-1, -1)})
    monkeypatch.setattr(memory, "worker_count", lambda: 8)
    monkeypatch.setattr(memory, "memory_gb", lambda: (20, 32, 30, 50))
    assert memory.contour_workers(["small"] * 5, 4) == 5
    assert memory.contour_workers(["small", "large"] * 5, 4) == 1
    assert memory.contour_workers(["small", "unknown"] * 5, 4) == 1
    monkeypatch.setattr(memory, "memory_gb", lambda: None)
    assert memory.contour_workers(["small"] * 5, 4) == 1


def test_image_header_reader_accepts_real_unicode_image_paths(tmp_path, monkeypatch):
    path = tmp_path / "테스트.png"
    ok, data = cv2.imencode(".png", np.zeros((30, 40), np.uint8))
    assert ok
    data.tofile(path)
    monkeypatch.setattr(memory, "worker_count", lambda: 8)
    monkeypatch.setattr(memory, "memory_gb", lambda: (5, 32, 10, 50))
    log = []
    assert memory.contour_workers([path] * 4, 4, log.append) == 4
    assert "unknown image size" not in log[0]


class ImmediatePool:
    """Completed futures, retained only weakly here so the test can detect a growing results list."""

    def __init__(self):
        self.futures, self.arrays = [], []

    def submit(self, function, *args):
        future = Future()
        result = function(*args)
        self.futures.append(weakref.ref(future))
        self.arrays.append(weakref.ref(result))
        future.set_result(result)
        return future


def test_consumed_futures_and_arrays_do_not_accumulate():
    pool = ImmediatePool()
    arguments = ((k,) for k in range(100))
    with closing(memory.bounded_results(pool, lambda k: np.array([k]), arguments, 3)) as results:
        for k in range(100):
            result = next(results)
            assert result[0] == k
            assert len(pool.futures) <= k + 3  # no eager submission of the whole folder
            assert pool.futures[k]() is None  # the yielded data is no longer held by a Future
            assert sum(future() is not None for future in pool.futures) <= 2
            assert sum(array() is not None for array in pool.arrays) <= 3
        with pytest.raises(StopIteration):
            next(results)


def test_closing_after_first_result_cancels_pending_tasks_and_does_not_fill_queue_again():
    futures = []

    def submit(function, *args):
        future = Future()
        if not futures:
            future.set_result(function(*args))
        futures.append(future)
        return future

    pool = SimpleNamespace(submit=submit)
    with closing(memory.bounded_results(pool, lambda k: k, ((k,) for k in range(100)), 3)) as results:
        assert next(results) == 0
    assert len(futures) == 3 and all(f.cancelled() for f in futures[1:])


def test_failure_cancels_pending_work_while_traceback_is_retained():
    futures = []

    def submit(function, *args):
        future = Future()
        if not futures:
            future.set_exception(MemoryError("simulated allocation failure"))
        futures.append(future)
        return future

    with pytest.raises(MemoryError) as stopped:
        list(memory.bounded_results(SimpleNamespace(submit=submit), lambda: None, [()] * 100, 2))
    assert stopped.traceback and len(futures) == 2 and futures[1].cancelled()


def test_borrowed_pool_is_left_open_and_results_keep_input_order():
    with ThreadPoolExecutor(2) as pool:
        results = list(memory.bounded_results(pool, lambda k: k * k, ((k,) for k in range(20)), 2))
        assert results == [k * k for k in range(20)]
        assert pool.submit(lambda: "still open").result() == "still open"
