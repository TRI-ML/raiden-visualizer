"""State/action conventions: the curated table covers every source, and the API joins it with info.json names."""

from fastapi.testclient import TestClient

from raiden_viz import app as app_module
from raiden_viz import config, lerobot, sources

INFO = {"features": {
    "observation.state": {"dtype": "float32", "shape": [3], "names": ["x", "y", "z"]},
    "action": {"dtype": "float32", "shape": [3], "names": ["action"]},          # opaque placeholder
    "joint_state": {"dtype": "float32", "shape": [2], "names": {"axes": ["j0", "j1"]}},
    "observation.images.top": {"dtype": "video", "shape": [256, 320, 3]},
    "subtask": {"dtype": "string", "shape": [1]},
    "is_demo": {"dtype": "bool", "shape": [1]},
    "timestamp": {"dtype": "float32", "shape": [1]},
}}


def test_every_configured_source_has_a_convention():
    for spec in config.SOURCES:
        c = config.CONVENTIONS.get(spec["id"])
        assert c, f"no convention for {spec['id']}"
        assert c["frame"] and c["rate"] and c["verified"]
        for g in c["groups"]:
            assert g["key"] and g["dims"] >= 1
            assert all(len(row) == 4 for row in g["axes"]), g["key"]


def test_lowdim_features_keeps_real_names_and_drops_placeholders():
    assert lerobot.lowdim_features(INFO) == [
        {"key": "observation.state", "dtype": "float32", "dims": 3, "names": ["x", "y", "z"]},
        {"key": "action", "dtype": "float32", "dims": 3, "names": None},
        {"key": "joint_state", "dtype": "float32", "dims": 2, "names": ["j0", "j1"]},
    ]


def test_endpoint_joins_curated_and_info_json(monkeypatch):
    src = sources.LeRobotSource({"id": "droid", "label": "DROID", "kind": "lerobot", "bucket": "b", "prefix": "p"})
    monkeypatch.setattr(src, "list_tasks", lambda: ["droid_full"])
    monkeypatch.setattr(app_module, "_src", lambda sid: src)
    monkeypatch.setattr(app_module.s3, "get_json", lambda key, bucket=None: INFO)
    monkeypatch.setattr(app_module, "_FEATURES", {})
    r = TestClient(app_module.app).get("/api/sources/droid/conventions").json()
    assert r["curated"]["rate"] == "15 Hz"
    assert r["features_from"] == "s3://b/p/droid_full/meta/info.json"
    assert [f["key"] for f in r["features"]] == ["observation.state", "action", "joint_state"]
