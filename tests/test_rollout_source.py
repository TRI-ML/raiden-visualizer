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
