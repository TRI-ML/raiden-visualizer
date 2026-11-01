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
# Plain prose first: `sentence` is one normal sentence; the fields are short values for a key/value grid.
# None means "not documented" (rendered muted, never as a warning). Slot ranges live in the axis tables.

def _conv(sentence, state, action=None, frame=None, rotation=None, gripper=None, rate=None, docs=()):
    return {"sentence": sentence, "state": state, "action": action, "frame": frame, "rotation": rotation,
            "gripper": gripper, "rate": rate, "docs": list(docs)}


_RAIDEN = _conv(
    "Each step stores both arms' joint angles and gripper openings, plus the commanded targets for the next step.",
    state="six joint angles and one gripper opening per arm; the teleop leader arm's joints are stored alongside",
    action="absolute joint targets, six joints plus the gripper per arm",
    frame="joint space; each arm's joints in its own base frame", rotation=None,
    gripper="0 closed … 1 open (normalised aperture)", rate="30 Hz policy steps (60 Hz for teleop recordings)",
    docs=["yam-robot svo2_recorder.py", "raiden_viz/urdf/yam.urdf"])
_ABC = _conv(
    "Each step stores both arms' joint angles and gripper openings as recorded by the bimanual YAM station.",
    state="six joint angles per arm (the MCAP also carries a world-frame end-effector pose, not plotted)",
    action="leader-arm targets exist in the recording but are not shown here",
    frame="joint space; the end-effector pose in the MCAP is in the world frame", rotation=None,
    gripper="0 closed … 1 open", rate="30 Hz", docs=["abc-130k README (Data format)"])
_V3AR_DROID = _conv(
    "Each step stores the single Franka arm's end-effector pose and gripper opening.",
    state="end-effector position and rotation, then the gripper opening", action=None,
    frame="robot base frame", rotation="6D (first two rows of the rotation matrix)",
    gripper="0 closed … 1 open", rate="15 Hz", docs=["flex-pi_release motion_check/cache_droid.py"])
_V3AR_REALOMNI = _conv(
    "Each step stores the two hand-held UMI grippers' poses and openings, left then right.",
    state="per gripper: position, rotation and opening fraction", action=None,
    frame=None, rotation="6D (first two rows of the rotation matrix)",
    gripper="0 closed … 1 open", rate="30 Hz", docs=["realomin_v3ar info.json", "flex-pi_release unwebbed_pools_REPORT.md"])
_V3AR_HABIT = _conv(
    "Each step stores both Franka arms' end-effector poses and gripper openings (layout inferred from the sibling exports).",
    state="per arm: position, rotation, gripper opening", action=None, frame=None,
    rotation="6D (first two rows of the rotation matrix)", gripper="0 closed … 1 open", rate="10 Hz",
    docs=["habit_v3ar info.json (generic names only)"])
_V3AR_AGIBOT = _conv(
    "Each step stores both arms' end-effector poses, then the two gripper openings.",
    state="left pose, right pose, then left and right gripper openings", action=None,
    frame="robot base frame", rotation="6D (first two columns of the rotation matrix)",
    gripper="0 closed … 1 open", rate="30 Hz", docs=["flex-pi_release docs/CORPUS_CONTRACT.md"])
_WEBBED = _conv(
    "Each step stores the two arms' end-effector poses, gripper openings and joint angles.",
    state="per arm: end-effector position and rotation; then both gripper openings; then six joint angles per arm",
    action="the same quantities for the next step (whether stored absolute or relative is not documented)",
    frame="each arm's own base frame: +x away from the base, +y to the robot's left, +z up",
    rotation="6D (first two rows of the rotation matrix)", gripper="0 closed … 1 open", rate="30 Hz",
    docs=["flex-pi_release docs/CORPUS_CONTRACT.md", "motion_check/proprio.py"])
_WORLDENGINE = _conv(
    "Each step stores both arms' joint angles and gripper openings, named per joint.",
    state="per arm: waist, shoulder, elbow, forearm roll, wrist angle, wrist rotate, then the gripper",
    action="the same channels as targets (absolute or delta is not documented)", frame="joint space",
    rotation=None, gripper="0 closed … 1 open", rate="60 Hz", docs=["WorldEngine meta/info.json"])
_MOLMO = _conv(
    "Each step stores both arms' six joint angles and gripper openings.",
    state="per arm: joints 0 to 5, then the gripper", action="absolute joint targets, same channels",
    frame="joint space", rotation=None, gripper="0 closed … 1 open", rate="30 Hz",
    docs=["MolmoAct2-BimanualYAM meta/info.json"])
_YAM_SIM = _conv(
    "Each step stores a 14-value state and action whose order is not documented (presumably six joints and a gripper per arm).",
    state=None, action=None, frame=None, rotation=None, gripper=None, rate="30 Hz",
    docs=["tri-yam sim_datasets lerobot meta/info.json"])

CONVENTIONS = {
    "raiden": _RAIDEN, "raiden2": _RAIDEN, "yam_russet": _RAIDEN, "policy_rollouts": _RAIDEN, "eval_rollouts": _RAIDEN,
    "yam": _ABC, "abc130k": _ABC, "abc130k_val": _ABC, "xdof_zed": _ABC,
    "droid": _V3AR_DROID, "realomni": _V3AR_REALOMNI, "habit": _V3AR_HABIT,
    "agibot26": _V3AR_AGIBOT, "agibot_beta": _V3AR_AGIBOT,
    "webbed": _WEBBED, "worldengine": _WORLDENGINE, "molmoact2_yam": _MOLMO, "yam_sim": _YAM_SIM,
}

_UNDOCUMENTED = _conv("The state convention of this dataset is not documented.", None)

# the dataset's representative state vector (signal key, width, names) for the overview's axis table
REALOMNI_NAMES = [f"{a}_{n}" for a in ("left", "right") for n in ("tcp_x", "tcp_y", "tcp_z", *[f"rot6d_{k}" for k in range(6)], "fraction_open")]
WE_NAMES = [f"{a}_{n}" for a in ("left", "right") for n in (*WE_JOINT_NAMES, "gripper")]
MOLMO_NAMES = [f"{a}_{n}" for a in ("left", "right") for n in (*[f"joint_{k}.pos" for k in range(6)], "gripper.pos")]
STATE_VECTOR = {
    **{s: [("follower_l_joint_pos_7d", 7, None), ("follower_r_joint_pos_7d", 7, None)] for s in ("raiden", "raiden2", "yam_russet", "policy_rollouts", "eval_rollouts")},
    **{s: [("left_arm_state", 6, None), ("left_ee_state", 1, None), ("right_arm_state", 6, None), ("right_ee_state", 1, None)] for s in ("yam", "abc130k", "abc130k_val", "xdof_zed")},
    "droid": [("joint_pos", 10, None)], "habit": [("joint_pos", 20, None)], "agibot26": [("joint_pos", 20, None)], "agibot_beta": [("joint_pos", 20, None)],
    "realomni": [("l_joint_pos", 10, REALOMNI_NAMES[:10]), ("r_joint_pos", 10, REALOMNI_NAMES[10:])],
    "webbed": [("state", 32, None)],
    "worldengine": [("l_joint_pos", 6, WE_NAMES[:6]), ("l_gripper_pos", 1, None), ("r_joint_pos", 6, WE_NAMES[7:13]), ("r_gripper_pos", 1, None)],
    "molmoact2_yam": [("l_joint_pos", 6, MOLMO_NAMES[:6]), ("l_gripper_pos", 1, None), ("r_joint_pos", 6, MOLMO_NAMES[7:13]), ("r_gripper_pos", 1, None)],
    "yam_sim": [("joint_pos", 14, None)],
}


def axis_groups(sid: str, vectors) -> list[dict]:
    """[{title, axes}] grouped per arm and quantity (the same split the plots use), for the axis tables."""
    out = []
    for key, dims, names in vectors:
        ax = axes(sid, key, dims, names)
        for g in groups(sid, key, dims, names):
            title = " ".join(x for x in (g["arm"], g["kind"]) if x) or key
            if g["kind"] == "other":
                title = key
            out.append({"title": title, "axes": [a for a in ax if a["i"] in set(g["idx"])]})
    return out


def convention(sid: str) -> dict:
    """The overview's card: prose + key/value fields + the collapsed axis table."""
    c = CONVENTIONS.get(sid, _UNDOCUMENTED)
    out = dict(c)
    out["axis_groups"] = axis_groups(sid, STATE_VECTOR.get(sid, []))
    out["n_axes"] = sum(len(g["axes"]) for g in out["axis_groups"])
    return out


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
    if "joint" in n or "revolute" in m or any(w in n for w in ("waist", "shoulder", "elbow", "forearm", "wrist")):
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
