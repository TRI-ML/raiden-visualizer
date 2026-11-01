"""Argus label tables (Zheyuan Hu's campaign mirror) joined to viewer episodes by uuid stem or LeRobot index."""
import io

import pyarrow as pa
import pyarrow.parquet as pq

from raiden_viz import argus, cache


def _table_bytes():
    ann = lambda src, cls, s, e, text: {"level": 1, "text": text, "start": s, "end": e, "weight": 1.0, "given": False,
                                         "source": src, "label_class": cls}
    rows = [{"episode_index": 0, "stem": "aaaa-1111", "annotations": [
                ann("task_string", "whole_task", 0, 300, "fold"), ann("argus:sol61", "subtask", 0, 90, "reach"),
                ann("argus:sol61", "subtask", 90, 300, "fold the shirt"), ann("argus:opus55", "subtask", 0, 300, "x"),
                ann("argus:sol61:dense", "subtask", 0, 30, "dense")]},
            {"episode_index": 1, "stem": "bbbb-2222", "annotations": [ann("task_string", "whole_task", 0, 10, "fold")]}]
    buf = io.BytesIO(); pq.write_table(pa.Table.from_pylist(rows), buf); return buf.getvalue()


def test_join_by_stem_and_by_index_only_sol_subtasks(monkeypatch, tmp_path):
    monkeypatch.setattr(cache, "get_json", lambda name, remote=False: None)
    monkeypatch.setattr(cache, "put_json", lambda name, value, remote=False: None)
    monkeypatch.setattr(argus.s3, "list_dirs", lambda prefix, bucket=None: ["abc130k_t007_fold_shirt", "abc130k_t008_other"])
    monkeypatch.setattr(argus.s3, "list_keys", lambda prefix, bucket=None, suffix=None: [type("O", (), {"key": prefix + "/chunk-000/file-000.parquet"})()])
    monkeypatch.setattr(argus.s3, "get_bytes", lambda key, bucket=None: _table_bytes())
    argus._repos.clear(); argus._tables.clear()
    abc = {"id": "abc_t", "argus": {"tables": "p/tables", "repo_prefix": "abc130k_t"}}
    got = argus.annotations_for(abc, "fold_shirt", "episode_aaaa-1111")
    assert [(a["t"], a["text"]) for a in got] == [(0.0, "reach"), (3.0, "fold the shirt")]   # 30 fps, Sol sub-tasks only
    assert argus.annotations_for(abc, "fold_shirt", "episode_bbbb-2222") == []                # whole-task rows don't count
    assert argus.annotations_for(abc, "unknown_task", "episode_aaaa-1111") == []
    assert argus.coverage(abc, "fold_shirt", ["episode_aaaa-1111", "episode_bbbb-2222"]) == 1
    molmo = {"id": "molmo_t", "argus": {"tables": "p/tables", "repo": "molmo_yam_v3", "join": "index"}}
    assert argus.annotations_for(molmo, "any task", "episode_000000")[0]["text"] == "reach"
    assert argus.coverage(molmo, "any task", ["episode_000000", "episode_000001", "episode_000009"]) == 1
    assert argus.annotations_for({"id": "x"}, "t", "e") == [] and argus.coverage({"id": "x"}, "t", []) is None
