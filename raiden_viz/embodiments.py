"""What the robot state / action vectors mean, per dataset.

Every entry cites where the convention is written down; where a source documents nothing we say
"not documented" rather than guess (the user fills those in). Served on the episode detail as
``robot.joint_legend`` (per plotted signal: index → name, meaning, unit) and on the overview as
``state_convention`` (one-line summary + the same axis lists), so the frontend only renders.

Sources of truth
  YAM joints      raiden_viz/urdf/yam.urdf (the file fk.py chains): joint names, rotation axis in the arm
                  base frame at q = 0, limits.
  raiden npz      yam-robot src/yam_robot/vibe/svo2_recorder.py (_write_robot_data): *_joint_pos_7d = 6 arm
                  joints + gripper, *_joint_cmd = commanded 7, *_joint_pos / *_gripper_pos split out;
                  gripper 0 = closed, 1 = open (yam_robot/intervention/leader_arm.py grip_clutch).
  ABC-130k MCAP   abc-130k README (vla_foundry_datasets_test/raw_datasets_bot/abc-130k/README.md): RobotState
                  = 6 joints in radians (+ a 4x4 world-frame EE pose, not plotted), GripperState 0 = closed,
                  1 = open; XDOF yam_raw uses the same messages on -proprio topics (raiden_viz/yam.py).
  eef32           flex-pi_release docs/CORPUS_CONTRACT.md §1 (branch argus-subgoal-labeling): the 32-D layout
                  of the webbed YAM corpora; rot6d = first two ROWS of R; arm base frame +x away from the base,
                  +y robot-left, +z up (argus_subgoal/ops/lang_v1/investigation/motion_check/proprio.py).
  v3ar pools      RealOmni names from its own meta/info.json; DROID from motion_check/cache_droid.py
                  ([xyz 3, rot6d 6, gripper opening 1 = open]); AgiBot [0:20] from CORPUS_CONTRACT.md (the 34-D
                  layout: rot6d stored as the first two COLUMNS of R).
  LeRobot sets    each repo's meta/info.json ``features.observation.state.names`` (WorldEngine, MolmoAct2).
"""

import xml.etree.ElementTree as ET

import numpy as np

from . import fk

RAD, M, FRAC = "rad", "m", "0–1"


def _yam_joints() -> list[dict]:
    """YAM joint table from the URDF: name, rotation axis expressed in the arm base frame at q = 0, limits."""
    root = ET.parse(fk._URDF).getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    by_parent = {j.find("parent").get("link"): (n, j) for n, j in joints.items()}
    out, link, R = [], "base_link", np.eye(3)
    while link in by_parent:
        n, j = by_parent[link]
        o = j.find("origin")
        rpy = [float(v) for v in o.get("rpy", "0 0 0").split()] if o is not None else [0, 0, 0]
        R = R @ fk._rpy_to_R(rpy)
        w = R @ np.array([float(v) for v in j.find("axis").get("xyz").split()])
        k = int(np.argmax(np.abs(w)))
        sign = "+" if w[k] > 0 else "-"
        lim = j.find("limit")
        out.append({"name": n, "meaning": f"revolute, about {sign}{'XYZ'[k]} of the arm base (q = 0)",
                    "unit": RAD, "range": [float(lim.get("lower")), float(lim.get("upper"))] if lim is not None else None})
        link = j.find("child").get("link")
    return out


YAM_JOINTS = _yam_joints()
YAM_GRIPPER = {"name": "gripper", "meaning": "parallel-jaw opening, 0 = closed, 1 = open", "unit": FRAC, "range": [0.0, 1.0]}

# the six YAM joint names the public LeRobot exports use (WorldEngine info.json), in joint order
WE_JOINT_NAMES = ["waist", "shoulder", "elbow", "forearm_roll", "wrist_angle", "wrist_rotate"]


def _ax(i, name, meaning, unit, rng=None):
    d = {"i": i, "name": name, "meaning": meaning, "unit": unit}
    if rng:
        d["range"] = rng
    return d


def _yam6(prefix="", unit=RAD, what="position"):
    return [_ax(i, f"{prefix}{j['name']}", f"{what}: {j['meaning']}", unit, j["range"] if unit == RAD else None)
            for i, j in enumerate(YAM_JOINTS)]


def _tcp_block(i0, arm, rot_rows=True):
    rot = "rot6d, first two ROWS of R" if rot_rows else "rot6d, first two COLUMNS of R"
    p = f"{arm}_" if arm else ""
    return ([_ax(i0 + k, f"{p}tcp_{c}", f"{arm} TCP position {c}".strip(), M) for k, c in enumerate("xyz")]
            + [_ax(i0 + 3 + k, f"{p}rot6d_{k}", f"{arm} TCP orientation ({rot})".strip(), "") for k in range(6)])


def _grip(i, arm, label="fraction_open"):
    return _ax(i, f"{arm}_{label}", f"{arm} gripper opening, 0 = closed, 1 = open", FRAC, [0.0, 1.0])


# ---- per-dataset conventions -------------------------------------------------------------------------

_RAIDEN = {
    "summary": "Joint space. Per arm: 6 YAM joint angles (rad) + gripper opening (0 closed … 1 open); "
               "`*_joint_pos_7d` / `*_joint_cmd` = the same 6 joints + gripper, `leader_*` = the teleop leader arm.",
    "frame": "arm base frame per arm (URDF yam.urdf); gripper is a normalised aperture",
    "action": "`*_joint_cmd`: absolute commanded joint targets (7 = 6 joints + gripper), same units as the state",
    "rate": "from `timestamps` (ns) in robot_data.npz, typically 30 Hz policy ticks / 60 Hz teleop",
    "docs": ["yam-robot svo2_recorder.py _write_robot_data", "raiden_viz/urdf/yam.urdf"],
}
_ABC = {
    "summary": "Joint space (MCAP RobotState): 6 YAM joint angles in radians per arm, plus a world-frame 4x4 "
               "end-effector pose (not plotted); GripperState aperture 0 = closed, 1 = open.",
    "frame": "joints in the arm's own frame; the EE pose in the MCAP is world-frame",
    "action": "leader / action topics exist in the MCAP but are not plotted here",
    "rate": "per-stream timestamps (absolute ns); typically 30 Hz",
    "docs": ["abc-130k README.md (Data format)"],
}
_V3AR_DROID = {
    "summary": "End-effector space, 10-D: [TCP xyz (m), rot6d (6, first two rows of R), gripper opening (1 = open)] "
               "for the single Franka Panda arm.",
    "frame": "robot base frame (DROID LeRobot port)", "action": "same 10-D layout; absolute vs delta: not documented in the export",
    "rate": "15 Hz", "docs": ["flex-pi_release motion_check/cache_droid.py"],
}
_V3AR_REALOMNI = {
    "summary": "End-effector space, 20-D: per hand-held UMI gripper [tcp_x, tcp_y, tcp_z (m), rot6d_0…5 (first two rows "
               "of R), fraction_open (0–1)], left then right (names in meta/info.json).",
    "frame": "not documented in the export (hand-held UMI grippers; ego/camera frame vs world is not stated)",
    "action": "same 20 names; absolute vs delta: not documented", "rate": "30 Hz",
    "docs": ["realomin_v3ar/*/meta/info.json feature names", "flex-pi_release unwebbed_pools_REPORT.md §1"],
}
_V3AR_HABIT = {
    "summary": "End-effector space, 20-D, two Franka Panda arms: [TCP xyz, rot6d (6), gripper] per arm "
               "(layout inferred from the same v3ar converter as DROID/RealOmni — not documented for HABIT).",
    "frame": "not documented", "action": "same 20-D layout; absolute vs delta: not documented", "rate": "10 Hz",
    "docs": ["habit_v3ar/habit_full/meta/info.json (generic names only)"],
}
_V3AR_AGIBOT = {
    "summary": "End-effector space, 20-D: [L TCP xyz, L rot6d (first two COLUMNS of R), R TCP xyz, R rot6d, "
               "L gripper, R gripper] — AgiBot's disk layout; the 7-joint channels of the 34-D layout are not in this export.",
    "frame": "robot base frame", "action": "same 20-D layout; absolute vs delta: not documented", "rate": "30 Hz",
    "docs": ["flex-pi_release docs/CORPUS_CONTRACT.md §1 (34-D layout, [0:20])"],
}
_WEBBED = {
    "summary": "eef32: [0:3] L TCP xyz (m), [3:9] L rot6d (first two rows of R), [9:12] R TCP xyz, [12:18] R rot6d, "
               "[18] L gripper, [19] R gripper (0–1, 1 = open), [20:26] L joints ×6 (rad), [26:32] R joints ×6.",
    "frame": "each arm in its own base frame: +x away from the base, +y robot-left, +z up",
    "action": "same 32 slots (`action`); the stored file's absolute/relative convention is not documented "
              "(Flex-π makes it body-frame relative at training time)",
    "rate": "30 Hz", "docs": ["flex-pi_release docs/CORPUS_CONTRACT.md §1", "motion_check/proprio.py"],
}
_WORLDENGINE = {
    "summary": "Joint space, 14-D: per arm waist, shoulder, elbow, forearm_roll, wrist_angle, wrist_rotate (rad) + gripper, "
               "left then right (names in meta/info.json).",
    "frame": "joint space", "action": "same 14 names; absolute vs delta: not documented in info.json", "rate": "60 Hz",
    "docs": ["yam_public/WorldEngine/*/meta/info.json"],
}
_MOLMO = {
    "summary": "Joint space, 14-D: per arm joint_0…joint_5 (rad) + gripper, left then right (names in meta/info.json; "
               "bi_yam_follower).",
    "frame": "joint space", "action": "same 14 names (`*.pos`); absolute joint targets", "rate": "30 Hz",
    "docs": ["yam_public/MolmoAct2-BimanualYAM/meta/info.json"],
}
_YAM_SIM = {
    "summary": "14-D state / action with generic names (`state`, `action`) — per-dimension order is not documented "
               "in the repo's info.json (yam_bimanual; presumably 6 joints + gripper per arm).",
    "frame": "not documented", "action": "not documented", "rate": "30 Hz",
    "docs": ["tri-yam/sim_datasets/*/lerobot/meta/info.json"],
}

CONVENTIONS = {
    "raiden": _RAIDEN, "raiden2": _RAIDEN, "yam_russet": _RAIDEN, "policy_rollouts": _RAIDEN, "eval_rollouts": _RAIDEN,
    "yam": _ABC, "abc130k": _ABC, "abc130k_val": _ABC, "xdof_zed": _ABC,
    "droid": _V3AR_DROID, "realomni": _V3AR_REALOMNI, "habit": _V3AR_HABIT,
    "agibot26": _V3AR_AGIBOT, "agibot_beta": _V3AR_AGIBOT,
    "webbed": _WEBBED, "worldengine": _WORLDENGINE, "molmoact2_yam": _MOLMO, "yam_sim": _YAM_SIM,
}

_UNDOCUMENTED = {"summary": "State convention not documented for this dataset.", "frame": "not documented",
                 "action": "not documented", "rate": None, "docs": []}


def convention(sid: str) -> dict:
    c = CONVENTIONS.get(sid, _UNDOCUMENTED)
    return {"summary": c["summary"], "frame": c["frame"], "action": c["action"], "rate": c["rate"], "docs": c["docs"]}


# ---- per-signal axis lists -----------------------------------------------------------------------------

def _arm(key: str) -> str:
    if key.startswith(("l_", "left")) or "_l_" in key:
        return "left"
    if key.startswith(("r_", "right")) or "_r_" in key:
        return "right"
    return ""


def _yam_signal(key: str, dims: int, names: list | None) -> list[dict] | None:
    """YAM joint-space signals: raiden npz keys, ABC/XDOF MCAP topics, LeRobot splits."""
    unit, what = RAD, "position"
    if "vel" in key:
        unit, what = "rad/s", "velocity"
    elif "eff" in key:
        unit, what = "effort (units not documented)", "effort"
    elif "cmd" in key:
        what = "commanded target"
    if "gripper" in key or "ee_state" in key or "eef" in key:
        return [dict(YAM_GRIPPER, i=0, meaning=f"{what}: {YAM_GRIPPER['meaning']}")]
    if names and len(names) == dims:
        return [_ax(i, n, f"{what} ({n})", unit) for i, n in enumerate(names)]
    if dims == 6:
        return _yam6(unit=unit, what=what)
    if dims == 7:
        return _yam6(unit=unit, what=what) + [dict(YAM_GRIPPER, i=6, meaning=f"{what}: {YAM_GRIPPER['meaning']}")]
    return None


def axes(sid: str, key: str, dims: int, names: list | None = None) -> list[dict]:
    """index → name/meaning/unit for one plotted signal; falls back to the dataset's own names, then `dim k`."""
    c = CONVENTIONS.get(sid)
    out = None
    if c is _WEBBED and dims == 32:
        out = (_tcp_block(0, "left") + _tcp_block(9, "right") + [_grip(18, "left", "grip"), _grip(19, "right", "grip")]
               + [_ax(20 + i, f"left_{j['name']}", f"left joint {j['meaning']}", RAD) for i, j in enumerate(YAM_JOINTS)]
               + [_ax(26 + i, f"right_{j['name']}", f"right joint {j['meaning']}", RAD) for i, j in enumerate(YAM_JOINTS)])
    elif c is _V3AR_DROID and dims == 10:
        out = _tcp_block(0, "") + [_ax(9, "gripper_opening", "gripper opening, 0 = closed, 1 = open", FRAC, [0.0, 1.0])]
    elif c is _V3AR_HABIT and dims == 20:
        out = _tcp_block(0, "left") + [_grip(9, "left")] + _tcp_block(10, "right") + [_grip(19, "right")]
    elif c is _V3AR_AGIBOT and dims == 20:
        out = _tcp_block(0, "left", rot_rows=False) + _tcp_block(9, "right", rot_rows=False) + [_grip(18, "left", "grip"), _grip(19, "right", "grip")]
    elif c is _V3AR_REALOMNI and names and len(names) == dims:
        out = [_ax(i, n, f"{n.replace('_', ' ')} ({M if 'tcp' in n else FRAC if 'open' in n else 'rot6d'})",
                   M if "tcp" in n else FRAC if "open" in n else "") for i, n in enumerate(names)]
    elif c in (_RAIDEN, _ABC, _WORLDENGINE, _MOLMO, _YAM_SIM):
        out = _yam_signal(key, dims, names if c is not _YAM_SIM else None)
    if out is None:
        out = [_ax(i, (names[i] if names and len(names) == dims else f"dim {i}"), "not documented", "") for i in range(dims)]
    arm = _arm(key)
    if arm and c in (_RAIDEN, _ABC, _WORLDENGINE, _MOLMO):
        for a in out:
            a["name"] = f"{arm} {a['name']}" if not a["name"].startswith(arm) else a["name"]
    return out


def joint_legend(sid: str, robot: dict | None) -> dict | None:
    """The episode-level legend: {summary, frame, action, rate, docs, signals: {key: [axes]}}."""
    if not robot or not robot.get("signals"):
        return None
    sig = {k: axes(sid, k, int(v.get("dims") or 0), v.get("names")) for k, v in robot["signals"].items()}
    return {**convention(sid), "signals": sig}


# ---- grouping for the trajectory plots ---------------------------------------------------------------------

ROWS = ("position", "rotation", "gripper", "joints", "other")


def _kind(a: dict) -> str:
    n, m = a["name"].lower(), a["meaning"].lower()
    if "tcp_" in n or "tcp position" in m:
        return "position"
    if "rot6d" in n or "rot6d" in m:
        return "rotation"
    if "grip" in n or "fraction_open" in n or "gripper" in m:
        return "gripper"
    if "joint" in n or "revolute" in m:
        return "joints"
    return "other"


def groups(sid: str, key: str, dims: int, names: list | None = None) -> list[dict]:
    """Split one plotted signal into (arm × quantity) panels: [{arm, kind, idx, rot}] in index order.
    `rot` says how a rot6d block encodes R ("rows" | "columns"), read off the axis meaning."""
    ax = axes(sid, key, dims, names)
    out: list[dict] = []
    for a in ax:
        nm = a["name"].lower()
        arm = "left" if nm.startswith("left") else "right" if nm.startswith("right") else _arm(key)
        kind = _kind(a)
        g = out[-1] if out and out[-1]["arm"] == arm and out[-1]["kind"] == kind else None
        if g is None:
            g = {"arm": arm, "kind": kind, "idx": []}
            if kind == "rotation":
                g["rot"] = "columns" if "COLUMNS" in a["meaning"] else "rows"
            out.append(g)
        g["idx"].append(a["i"])
    return out


def rot6d_to_matrix(r6, rows: bool = True) -> np.ndarray:
    """(N, 6) → (N, 3, 3). The 6 numbers are the first two rows (UMI/Drake row convention) or the first two
    columns of R; Gram–Schmidt gives the third."""
    r6 = np.asarray(r6, dtype=np.float64).reshape(-1, 6)
    a, b = r6[:, 0:3], r6[:, 3:6]
    a = a / np.maximum(np.linalg.norm(a, axis=1, keepdims=True), 1e-9)
    b = b - np.sum(a * b, axis=1, keepdims=True) * a
    b = b / np.maximum(np.linalg.norm(b, axis=1, keepdims=True), 1e-9)
    c = np.cross(a, b)
    R = np.stack([a, b, c], axis=1)          # rows
    return R if rows else np.transpose(R, (0, 2, 1))


def matrix_to_euler_xyz_deg(R: np.ndarray) -> np.ndarray:
    """Intrinsic x-y-z (roll, pitch, yaw) in degrees from (N, 3, 3): R = Rx(roll) · Ry(pitch) · Rz(yaw)."""
    pitch = np.arcsin(np.clip(R[:, 0, 2], -1.0, 1.0))
    roll = np.arctan2(-R[:, 1, 2], R[:, 2, 2])
    yaw = np.arctan2(-R[:, 0, 1], R[:, 0, 0])
    return np.degrees(np.stack([roll, pitch, yaw], axis=1))


def rot6d_to_euler_deg(r6, rows: bool = True) -> np.ndarray:
    return matrix_to_euler_xyz_deg(rot6d_to_matrix(r6, rows))


def layout(sid: str, robot: dict) -> list[dict] | None:
    """Panels for the trajectory grid, adding derived roll/pitch/yaw signals for every rot6d block:
    [{arm, row, key, idx, raw_key, raw_idx, label}] ordered position → rotation → gripper → joints.
    Returns None when nothing splits (a dataset with a single undocumented vector)."""
    panels: list[dict] = []
    for key, sig in list(robot.get("signals", {}).items()):
        if key.endswith("__rpy"):
            continue
        gs = groups(sid, key, int(sig.get("dims") or 0), sig.get("names"))
        if len(gs) <= 1 and gs and gs[0]["kind"] == "other":
            return None if len(robot["signals"]) == 1 else panels or None
        for g in gs:
            panel = {"arm": g["arm"], "row": g["kind"], "key": key, "idx": g["idx"],
                     "label": f"{g['arm']} {g['kind']}".strip()}
            if g["kind"] == "rotation" and len(g["idx"]) == 6:
                series = np.asarray(sig["series"], dtype=np.float64)[:, g["idx"]]
                eul = rot6d_to_euler_deg(series, rows=g.get("rot") != "columns")
                dk = f"{key}__{g['arm'] or 'ee'}__rpy"
                robot["signals"][dk] = {"dims": 3, "series": np.round(eul, 3).tolist(), "min": round(float(eul.min()), 3),
                                        "max": round(float(eul.max()), 3), "names": ["roll", "pitch", "yaw"], "derived": True}
                panel.update({"raw_key": key, "raw_idx": g["idx"], "key": dk, "idx": [0, 1, 2],
                              "label": f"{g['arm']} rotation (roll / pitch / yaw, intrinsic xyz, deg)".strip()})
            panels.append(panel)
    order = {r: i for i, r in enumerate(ROWS)}
    panels.sort(key=lambda p: (order[p["row"]], {"left": 0, "": 1, "right": 2}[p["arm"]], p["key"]))
    return panels or None
