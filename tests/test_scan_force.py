"""POST /scan?force=true restarts a FINISHED scan (a scan run before a dataset was
uploaded is complete and wrong); it never interrupts a running one."""

import threading

from raiden_viz import sources


class _Src(sources.Source):
    def __init__(self):
        super().__init__({"id": "t", "label": "T", "kind": "raiden", "bucket": "b", "prefix": "p"})
        self.pairs = []

    def _stat_pairs(self, full=False):
        return list(self.pairs), len(self.pairs)

    def _safe_stat(self, task, episode):
        return {"task": task, "episode": episode}


def _wait_done(src):
    for _ in range(200):
        snap = src.scan_snapshot()
        if snap and snap["done"]:
            return snap
        threading.Event().wait(0.01)
    raise AssertionError("scan did not finish")


def test_force_rescans_a_finished_scan(monkeypatch):
    monkeypatch.setattr(sources, "_SCANS", {})
    src = _Src()
    src.scan_start()
    assert _wait_done(src)["total_episodes"] == 0
    src.pairs = [("task", "ep0"), ("task", "ep1")]
    assert src.scan_start()["total_episodes"] == 0, "without force the stale scan is reused"
    src.scan_start(force=True)
    assert _wait_done(src)["total_episodes"] == 2


def test_force_does_not_interrupt_a_running_scan(monkeypatch):
    monkeypatch.setattr(sources, "_SCANS", {})
    src = _Src()
    st = {"running": True, "done": False, "total": 5, "episodes": [], "error": None,
          "lock": threading.Lock()}
    sources._SCANS[src._scan_id()] = st
    snap = src.scan_start(force=True)
    assert snap["running"] and snap["total_episodes"] == 5
    assert sources._SCANS[src._scan_id()] is st
