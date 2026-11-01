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
        assert c["summary"] and c["docs"], s["id"]


def test_joint_legend_shape():
    robot = {"signals": {"follower_l_joint_pos": {"dims": 6}, "follower_l_gripper_pos": {"dims": 1}}}
    lg = emb.joint_legend("raiden", robot)
    assert set(lg["signals"]) == {"follower_l_joint_pos", "follower_l_gripper_pos"}
    assert len(lg["signals"]["follower_l_joint_pos"]) == 6 and lg["summary"].startswith("Joint space")
    assert emb.joint_legend("raiden", None) is None
