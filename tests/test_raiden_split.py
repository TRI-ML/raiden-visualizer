"""Raiden vs Raiden 2.0: the harvest era is read off the recording (camera_config + Task Hub name + date)."""
from raiden_viz import sources

NEW = {"task_name": "HangMugs", "timestamp": "2026-10-07T18:38:59", "camera_config": {"ego_camera": {}}}
OLD = {"task_name": "pickup_apple", "timestamp": "2026-04-08T10:33:00"}


def test_classifier_needs_all_three_signals():
    assert sources.is_harvest_task("HangMugs", NEW)
    assert not sources.is_harvest_task("pickup_apple", OLD)
    assert not sources.is_harvest_task("hang_mug_on_mug_tree", NEW)                      # snake_case: not a Task Hub task
    assert not sources.is_harvest_task("BlockOnBlockWithViz", {**NEW, "camera_config": None, "timestamp": "2026-06-29"})
    assert not sources.is_harvest_task("ArshPickBanana", {**NEW, "timestamp": "2026-07-12"})   # before the console landed
    assert not sources.is_harvest_task("HangMugs_dagger", None)                           # stray dir without metadata


def test_split_sources_partition_the_prefix(monkeypatch, cache_dir):
    tree = {"p": ["HangMugs", "pickup_apple", "HangMugs_dagger"], "p/HangMugs": ["e1", "e2"], "p/pickup_apple": ["e1"], "p/HangMugs_dagger": ["e1"]}
    md = {"p/HangMugs/e2/metadata.json": NEW, "p/pickup_apple/e1/metadata.json": OLD}
    monkeypatch.setattr(sources.s3, "list_dirs", lambda prefix, bucket=None: sorted(tree.get(prefix.rstrip("/"), [])))
    monkeypatch.setattr(sources.s3, "try_get_json", lambda key, bucket=None: md.get(key))
    legacy = sources.RaidenSource({"id": "raiden", "label": "R", "kind": "raiden", "bucket": "b", "prefix": "p", "split": "legacy"})
    harvest = sources.RaidenSource({"id": "raiden2", "label": "R2", "kind": "raiden", "bucket": "b", "prefix": "p", "split": "harvest"})
    assert harvest.list_tasks() == ["HangMugs"] and legacy.list_tasks() == ["HangMugs_dagger", "pickup_apple"]
    assert legacy.list_episodes("HangMugs") == [] and harvest.list_episodes("HangMugs") == ["e1", "e2"]
    # the classification is persisted per task with its episode count, so a second source re-reads no metadata
    monkeypatch.setattr(sources.s3, "try_get_json", lambda key, bucket=None: (_ for _ in ()).throw(AssertionError(key)))
    fresh = sources.RaidenSource({"id": "x", "label": "x", "kind": "raiden", "bucket": "b", "prefix": "p", "split": "harvest"})
    assert fresh.list_tasks() == ["HangMugs"]
