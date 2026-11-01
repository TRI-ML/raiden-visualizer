"""yam-eval policy rollouts: the raiden layout nested under <run>_<step>/<task>/rollout_NNNN (Vibe) and
<formal_id>/<run>_<step>/<task>/rollout_NNNN (ranked evals)."""
from raiden_viz import config, sources

TREE = {
    "tri-yam/policy_rollouts": ["pi05_teabag_19999", "cosmos3_toast_15000"],
    "tri-yam/policy_rollouts/pi05_teabag_19999": ["place_teabag"],
    "tri-yam/policy_rollouts/cosmos3_toast_15000": ["toast_bread", "place_teabag"],
    "tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag": ["rollout_0002", "rollout_0009"],
    "tri-yam/policy_rollouts/cosmos3_toast_15000/toast_bread": ["rollout_0001"],
    "tri-yam/policy_rollouts/cosmos3_toast_15000/place_teabag": ["rollout_0001"],
    "tri-yam/eval_rollouts": ["fe-20261031-105454-7ec4b0"],
    "tri-yam/eval_rollouts/fe-20261031-105454-7ec4b0": ["mugs_run0_1000"],
    "tri-yam/eval_rollouts/fe-20261031-105454-7ec4b0/mugs_run0_1000": ["hang_mugs"],
    "tri-yam/eval_rollouts/fe-20261031-105454-7ec4b0/mugs_run0_1000/hang_mugs": ["rollout_0001"],
}


def _fake_list_dirs(prefix, bucket=None):
    return sorted(TREE.get(prefix.rstrip("/"), []))


def _src(sid):
    spec = next(s for s in config.SOURCES if s["id"] == sid)
    return sources._KINDS[spec["kind"]](spec)   # get_sources() memoises process-wide; build the adapter directly


def test_vibe_rollouts_span_models_per_task(monkeypatch):
    monkeypatch.setattr(sources.s3, "list_dirs", _fake_list_dirs)
    src = _src("policy_rollouts")
    assert isinstance(src, sources.RolloutSource) and src.depth == 1
    assert src.list_tasks() == ["place_teabag", "toast_bread"]
    eps = src.list_episodes("place_teabag")
    assert eps == ["cosmos3_toast_15000::rollout_0001", "pi05_teabag_19999::rollout_0002", "pi05_teabag_19999::rollout_0009"]
    assert src._ep_prefix("place_teabag", eps[1]) == "tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag/rollout_0002"


def test_eval_rollouts_carry_the_eval_id(monkeypatch):
    monkeypatch.setattr(sources.s3, "list_dirs", _fake_list_dirs)
    src = _src("eval_rollouts")
    assert src.depth == 2 and src.list_tasks() == ["hang_mugs"]
    ep = src.list_episodes("hang_mugs")[0]
    assert ep == "fe-20261031-105454-7ec4b0::mugs_run0_1000::rollout_0001"
    assert src._ep_prefix("hang_mugs", ep) == "tri-yam/eval_rollouts/fe-20261031-105454-7ec4b0/mugs_run0_1000/hang_mugs/rollout_0001"


def test_empty_prefix_lists_nothing(monkeypatch):
    monkeypatch.setattr(sources.s3, "list_dirs", lambda prefix, bucket=None: [])
    assert _src("eval_rollouts").list_tasks() == []


# ---- eval sessions: rollouts referenced by eval_session.json, recordings in the Vibe tree ----
SESSION = {
    "eval_id": "20261029-221814-d77b", "task": "place_teabag", "task_label": "Steep Teabag", "operator": "nico",
    "model": "pi05_teabag @ 19999", "prompt": "Place the teabag into the teapot", "station": {"id": "taro"},
    "n": 3, "k": 1, "rate": 0.33,
    "rollouts": [
        {"run_id": "20261029-161924-aa86", "verdict": "failure"},                               # via History record
        {"run_id": "20261029-162932-9d0b", "verdict": "success",
         "recording_s3": "s3://b/tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag/rollout_0009/"},
        {"run_id": "20261029-170000-dead", "verdict": "failure"},                               # recording deleted
        {"run_id": "20261029-171111-0000", "verdict": None},                                    # unscored: ignored
    ],
}
HISTORY = {
    "tri-yam/vibe_history/taro/20261029-161924-aa86/record.json":
        {"recording_s3": "s3://b/tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag/rollout_0021/"},
    "tri-yam/vibe_history/taro/20261029-170000-dead/record.json":
        {"recording_s3": "s3://b/tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag/rollout_0030/"},
}
S_TREE = {"tri-yam/eval_sessions": ["taro"], "tri-yam/eval_sessions/taro": ["20261029-221814-d77b", "20261029-133743-9d42"]}
EXISTING = {  # metadata.json objects that exist (rollout_0030 was deleted)
    "tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag/rollout_0021/metadata.json",
    "tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag/rollout_0009/metadata.json",
}


def _eval_src(monkeypatch, cache_dir):
    monkeypatch.setattr(sources.s3, "list_dirs", lambda prefix, bucket=None: sorted(S_TREE.get(prefix.rstrip("/"), [])))
    monkeypatch.setattr(sources.s3, "try_get_json", lambda key, bucket=None: (
        SESSION if key.endswith("20261029-221814-d77b/eval_session.json") else HISTORY.get(key)))
    monkeypatch.setattr(sources.s3, "try_head", lambda key, bucket=None: object() if key in EXISTING else None)
    spec = {**next(s for s in config.SOURCES if s["id"] == "eval_rollouts"), "bucket": "b"}
    return sources.RolloutSource(spec)


def test_eval_sessions_list_their_saved_rollouts_only(monkeypatch, cache_dir):
    src = _eval_src(monkeypatch, cache_dir)
    assert src.list_tasks() == ["place_teabag"]          # the aborted session (no eval_session.json) adds nothing
    eps = src.list_episodes("place_teabag")
    assert eps == ["20261029-221814-d77b::pi05_teabag_19999::rollout_0009", "20261029-221814-d77b::pi05_teabag_19999::rollout_0021"]
    assert src._ep_prefix("place_teabag", eps[1]) == "tri-yam/policy_rollouts/pi05_teabag_19999/place_teabag/rollout_0021"
    # the History lookup is memoised on disk (a run's recording never moves)
    assert sources.cache.get_json("evalrec_eval_rollouts_20261029-161924-aa86.json")["recording_s3"].endswith("rollout_0021/")


def test_eval_session_detail_carries_the_sessions_verdict(monkeypatch, cache_dir):
    src = _eval_src(monkeypatch, cache_dir)
    monkeypatch.setattr(sources.s3, "get_json", lambda key, bucket=None: {"task_instruction": None, "status": "success", "duration_s": 6.4, "cameras": []})
    monkeypatch.setattr(sources.s3, "list_files", lambda prefix, bucket=None: [])
    d = src.episode_detail("place_teabag", "20261029-221814-d77b::pi05_teabag_19999::rollout_0021")
    es = d["metadata"]["eval_session"]
    assert es["eval_id"] == "20261029-221814-d77b" and es["operator"] == "nico" and es["verdict"] == "failure"
    assert d["status"] == "failure" and d["instruction"] == "Place the teabag into the teapot"
    assert src.episode_stat("place_teabag", "20261029-221814-d77b::pi05_teabag_19999::rollout_0021")["status"] == "failure"
