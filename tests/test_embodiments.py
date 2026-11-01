"""The per-dataset state / action legend (raiden_viz/embodiments.py)."""

from raiden_viz import config
from raiden_viz import embodiments as emb


def test_yam_joints_come_from_the_urdf():
    names = [j["name"] for j in emb.YAM_JOINTS]
    assert names == ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
    # base yaw first, tool roll last, limits read from the URDF
    assert "+Z" in emb.YAM_JOINTS[0]["meaning"] and "+X" in emb.YAM_JOINTS[5]["meaning"]
    assert emb.YAM_JOINTS[0]["range"] == [-2.61799, 3.13]


def test_raiden_signals_six_joints_plus_gripper():
    pos = emb.axes("raiden2", "follower_l_joint_pos", 6)
    assert [a["name"] for a in pos] == [f"left joint{i}" for i in range(1, 7)]
    assert all(a["unit"] == "rad" for a in pos)
    cmd = emb.axes("raiden", "follower_r_joint_cmd", 7)
    assert len(cmd) == 7 and cmd[6]["name"] == "right gripper" and cmd[6]["range"] == [0.0, 1.0]
    assert "commanded" in cmd[0]["meaning"]
    grip = emb.axes("yam_russet", "follower_l_gripper_pos", 1)
    assert grip[0]["unit"] == "0–1" and "0 = closed" in grip[0]["meaning"]


def test_franka_v3ar_layouts():
    droid = emb.axes("droid", "joint_pos", 10)
    assert [a["name"] for a in droid[:3]] == ["tcp_x", "tcp_y", "tcp_z"] and [a["unit"] for a in droid[:3]] == ["m"] * 3
    assert len(droid) == 10 and droid[9]["unit"] == "0–1" and "ROWS" in droid[3]["meaning"]
    habit = emb.axes("habit", "joint_cmd", 20)
    assert len(habit) == 20 and habit[9]["name"] == "left_fraction_open" and habit[19]["name"] == "right_fraction_open"
    agibot = emb.axes("agibot26", "joint_pos", 20)
    assert agibot[18]["name"] == "left_grip" and "COLUMNS" in agibot[3]["meaning"]


def test_webbed_eef32_and_lerobot_names():
    w = emb.axes("webbed", "state", 32)
    assert len(w) == 32 and w[18]["name"] == "left_grip" and w[20]["name"] == "left_joint1" and w[31]["name"] == "right_joint6"
    names = ["left_waist", "left_shoulder", "left_elbow", "left_forearm_roll", "left_wrist_angle", "left_wrist_rotate"]
    we = emb.axes("worldengine", "l_joint_pos", 6, names)
    assert [a["name"] for a in we] == names
    unknown = emb.axes("yam_sim", "joint_pos", 14)
    assert unknown[0]["name"] == "dim 0" and unknown[0]["meaning"] == "not documented"


def test_every_configured_source_has_a_convention():
    missing = [s["id"] for s in config.SOURCES if s["id"] not in emb.CONVENTIONS]
    assert missing == [], missing
    for s in config.SOURCES:
        c = emb.convention(s["id"])
        assert c["sentence"] and c["docs"] and c["n_axes"] > 0, s["id"]
        assert set(c) >= {"state", "action", "frame", "rotation", "gripper", "rate", "axis_groups"}


def test_joint_legend_shape():
    robot = {"signals": {"follower_l_joint_pos": {"dims": 6}, "follower_l_gripper_pos": {"dims": 1}}}
    lg = emb.joint_legend("raiden", robot)
    assert set(lg["signals"]) == {"follower_l_joint_pos", "follower_l_gripper_pos"}
    assert len(lg["signals"]["follower_l_joint_pos"]) == 6 and lg["sentence"].startswith("Each step")
    assert emb.joint_legend("raiden", None) is None


def test_rot6d_to_euler_known_rotations():
    import numpy as np
    # identity → 0, 0, 0 (rows and columns agree on the identity)
    assert np.allclose(emb.rot6d_to_euler_deg([[1, 0, 0, 0, 1, 0]]), 0, atol=1e-9)
    c, s = np.cos(np.pi / 2), np.sin(np.pi / 2)
    rz = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    assert np.allclose(emb.rot6d_to_euler_deg([rz[:2].ravel()]), [[0, 0, 90]], atol=1e-6)                 # rows
    assert np.allclose(emb.rot6d_to_euler_deg([rz[:, :2].T.ravel()], rows=False), [[0, 0, 90]], atol=1e-6)  # columns
    # 30° roll about x, intrinsic xyz: roll comes back as 30
    a = np.radians(30); rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
    assert np.allclose(emb.rot6d_to_euler_deg([rx[:2].ravel()]), [[30, 0, 0]], atol=1e-6)


def test_eef32_groups_and_layout():
    import numpy as np
    gs = emb.groups("webbed", "state", 32)
    assert [(g["arm"], g["kind"], len(g["idx"])) for g in gs] == [
        ("left", "position", 3), ("left", "rotation", 6), ("right", "position", 3), ("right", "rotation", 6),
        ("left", "gripper", 1), ("right", "gripper", 1), ("left", "joints", 6), ("right", "joints", 6)]
    assert gs[1]["rot"] == "rows" and emb.groups("agibot26", "joint_pos", 20)[1]["rot"] == "columns"
    robot = {"signals": {"state": {"dims": 32, "series": np.zeros((4, 32)).tolist(), "min": 0, "max": 0}}}
    robot["signals"]["state"]["series"] = [[0, 0, 0, 1, 0, 0, 0, 1, 0] + [0] * 9 + [0, 0] + [0] * 12 for _ in range(4)]
    lay = emb.layout("webbed", robot)
    rows = [(p["arm"], p["row"]) for p in lay]
    assert rows[:2] == [("left", "position"), ("right", "position")] and ("left", "rotation") in rows and ("right", "joints") in rows
    rot = next(p for p in lay if p["row"] == "rotation" and p["arm"] == "left")
    assert rot["key"] == "state__left__rpy" and rot["raw_key"] == "state" and rot["raw_idx"] == [3, 4, 5, 6, 7, 8]
    assert robot["signals"]["state__left__rpy"]["names"] == ["roll", "pitch", "yaw"]
    # a single undocumented vector has no split
    assert emb.layout("yam_sim", {"signals": {"joint_pos": {"dims": 14, "series": [[0] * 14], "min": 0, "max": 0}}}) is None


def test_convention_axis_groups_follow_the_plot_split():
    c = emb.convention("webbed")
    assert [g["title"] for g in c["axis_groups"]][:4] == ["left position", "left rotation", "right position", "right rotation"]
    assert c["n_axes"] == 32 and c["frame"].startswith("each arm")
    we = emb.convention("worldengine")
    assert [g["title"] for g in we["axis_groups"]] == ["left joints", "left gripper", "right joints", "right gripper"]
    nd = emb.convention("yam_sim")
    assert nd["state"] is None and nd["axis_groups"][0]["axes"][0]["meaning"] == "not documented"
