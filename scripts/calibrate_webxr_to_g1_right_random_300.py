import argparse
import json
import math
import random
import socket
import threading
import time
from pathlib import Path

import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_samples", type=int, default=300)
parser.add_argument("--out", type=str, default="/tmp/g1_webxr_right_random_calibration_300.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG


# ------------------------------------------------------------
# UDP WebXR receiver
# ------------------------------------------------------------

latest = {
    "packet": None,
    "time": 0.0,
    "count": 0,
}


def udp_server():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(("127.0.0.1", 8765))
    print("[UDP] listening on 127.0.0.1:8765")

    while True:
        try:
            data, _ = sock.recvfrom(65535)
            obj = json.loads(data.decode("utf-8"))
            latest["packet"] = obj
            latest["time"] = time.time()
            latest["count"] += 1

            if latest["count"] % 240 == 0:
                hands = []
                if isinstance(obj.get("hands"), dict):
                    hands = list(obj["hands"].keys())
                print("[UDP]", latest["count"], hands)
        except Exception as e:
            print("[UDP ERROR]", e)


threading.Thread(target=udp_server, daemon=True).start()


# ------------------------------------------------------------
# Input thread
# ------------------------------------------------------------

input_events = []


def input_thread():
    print("")
    print("INPUT CONTROLS:")
    print("  ENTER     = save current matched pose and advance")
    print("  s + ENTER = skip current pose")
    print("  w + ENTER = write/save JSON now, keep going")
    print("  q + ENTER = write/save JSON and quit")
    print("")
    while True:
        line = input().strip().lower()
        input_events.append(line)


threading.Thread(target=input_thread, daemon=True).start()


# ------------------------------------------------------------
# Quaternion / WebXR helpers
# ------------------------------------------------------------

def get_hand_packet(hand_name: str):
    pkt = latest["packet"]
    if not pkt:
        return None

    hands = pkt.get("hands")
    if isinstance(hands, dict):
        return hands.get(hand_name)

    # Fallback for flat packet style: {"right_wrist": {...}}
    flat = {}
    prefix = f"{hand_name}_"
    for k, v in pkt.items():
        if k.startswith(prefix):
            flat[k[len(prefix):]] = v

    return flat if flat else None


def get_joint(hand_name: str, joint_name: str):
    hand = get_hand_packet(hand_name)
    if not hand:
        return None

    j = hand.get(joint_name)
    if isinstance(j, dict) and "p" in j:
        return j

    return None


def webxr_pos_to_isaac(p):
    # Same mapping used in your working bimanual scripts:
    # WebXR [x,y,z] -> Isaac [-z,-x,y]
    x, y, z = p
    return torch.tensor([-z, -x, y], dtype=torch.float32)


def quat_normalize(q):
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def quat_inv(q):
    out = q.clone()
    out[..., 1:] *= -1.0
    return quat_normalize(out)


def quat_mul(q1, q2):
    # q format = [w,x,y,z]
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)

    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def quat_wxyz_to_matrix(q):
    q = quat_normalize(q)
    w, x, y, z = q.tolist()

    return torch.tensor(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=torch.float32,
    )


def matrix_to_quat_wxyz(m):
    m00 = float(m[0, 0])
    m01 = float(m[0, 1])
    m02 = float(m[0, 2])
    m10 = float(m[1, 0])
    m11 = float(m[1, 1])
    m12 = float(m[1, 2])
    m20 = float(m[2, 0])
    m21 = float(m[2, 1])
    m22 = float(m[2, 2])

    tr = m00 + m11 + m22

    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m21 - m12) / s
        y = (m02 - m20) / s
        z = (m10 - m01) / s
    elif m00 > m11 and m00 > m22:
        s = math.sqrt(1.0 + m00 - m11 - m22) * 2
        w = (m21 - m12) / s
        x = 0.25 * s
        y = (m01 + m10) / s
        z = (m02 + m20) / s
    elif m11 > m22:
        s = math.sqrt(1.0 + m11 - m00 - m22) * 2
        w = (m02 - m20) / s
        x = (m01 + m10) / s
        y = 0.25 * s
        z = (m12 + m21) / s
    else:
        s = math.sqrt(1.0 + m22 - m00 - m11) * 2
        w = (m10 - m01) / s
        x = (m02 + m20) / s
        y = (m12 + m21) / s
        z = 0.25 * s

    return quat_normalize(torch.tensor([w, x, y, z], dtype=torch.float32))


def webxr_quat_xyzw_to_isaac_wxyz(q_xyzw):
    """
    WebXR q = [qx,qy,qz,qw].
    Convert to Isaac basis using same axis transform as position:
      WebXR [x,y,z] -> Isaac [-z,-x,y]
    """
    qx, qy, qz, qw = q_xyzw
    q_web = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)

    r_web = quat_wxyz_to_matrix(q_web)

    a = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    r_isaac = a @ r_web @ a.T
    return matrix_to_quat_wxyz(r_isaac)


def estimate_webxr_palm_frame(hand_name: str):
    """
    Returns:
      palm_quat_isaac_wxyz
      palm_normal_isaac_xyz
      palm_axes_isaac: across/forward/normal
    """
    wrist = get_joint(hand_name, "wrist")
    index = get_joint(hand_name, "index-finger-metacarpal")
    middle = get_joint(hand_name, "middle-finger-metacarpal")
    pinky = get_joint(hand_name, "pinky-finger-metacarpal")

    if wrist is None or index is None or middle is None or pinky is None:
        return None, None, None

    pw = webxr_pos_to_isaac(wrist["p"])
    pi = webxr_pos_to_isaac(index["p"])
    pm = webxr_pos_to_isaac(middle["p"])
    pp = webxr_pos_to_isaac(pinky["p"])

    across = pi - pp
    across = across / torch.clamp(torch.linalg.norm(across), min=1e-6)

    forward = pm - pw
    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-6)

    normal = torch.cross(across, forward, dim=0)
    normal = normal / torch.clamp(torch.linalg.norm(normal), min=1e-6)

    forward = torch.cross(normal, across, dim=0)
    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-6)

    r = torch.stack([across, forward, normal], dim=1)
    q = matrix_to_quat_wxyz(r)

    axes = {
        "across_xyz": across.detach().cpu().tolist(),
        "forward_xyz": forward.detach().cpu().tolist(),
        "normal_xyz": normal.detach().cpu().tolist(),
    }

    return q, normal, axes


WEBXR_JOINT_NAMES = [
    "wrist",

    "thumb-metacarpal",
    "thumb-phalanx-proximal",
    "thumb-phalanx-distal",
    "thumb-tip",

    "index-finger-metacarpal",
    "index-finger-phalanx-proximal",
    "index-finger-phalanx-intermediate",
    "index-finger-phalanx-distal",
    "index-finger-tip",

    "middle-finger-metacarpal",
    "middle-finger-phalanx-proximal",
    "middle-finger-phalanx-intermediate",
    "middle-finger-phalanx-distal",
    "middle-finger-tip",

    "ring-finger-metacarpal",
    "ring-finger-phalanx-proximal",
    "ring-finger-phalanx-intermediate",
    "ring-finger-phalanx-distal",
    "ring-finger-tip",

    "pinky-finger-metacarpal",
    "pinky-finger-phalanx-proximal",
    "pinky-finger-phalanx-intermediate",
    "pinky-finger-phalanx-distal",
    "pinky-finger-tip",
]


def collect_webxr_keypoints(hand_name: str):
    """
    Save raw and Isaac-converted keypoints.
    """
    out = {}

    for name in WEBXR_JOINT_NAMES:
        j = get_joint(hand_name, name)
        if j is None:
            continue

        entry = {
            "p_raw_xyz": j["p"],
            "p_isaac_xyz": webxr_pos_to_isaac(j["p"]).detach().cpu().tolist(),
            "r": j.get("r", None),
        }

        if "q" in j:
            qx, qy, qz, qw = j["q"]
            entry["q_raw_xyzw"] = j["q"]
            entry["q_raw_wxyz"] = [qw, qx, qy, qz]
            entry["q_isaac_wxyz"] = webxr_quat_xyzw_to_isaac_wxyz(j["q"]).detach().cpu().tolist()

        out[name] = entry

    return out


# ------------------------------------------------------------
# Debug draw
# ------------------------------------------------------------

def setup_debug_draw():
    try:
        from isaacsim.util.debug_draw import _debug_draw
        draw = _debug_draw.acquire_debug_draw_interface()
        print("[VIZ] debug draw enabled")
        return draw
    except Exception as e:
        print("[VIZ] debug draw unavailable:", e)
        return None


def safe_clear_draw(draw):
    if draw is None:
        return
    try:
        draw.clear_points()
    except Exception:
        pass
    try:
        draw.clear_lines()
    except Exception:
        pass


HAND_BONES = [
    ("wrist", "thumb-metacarpal"),
    ("thumb-metacarpal", "thumb-phalanx-proximal"),
    ("thumb-phalanx-proximal", "thumb-phalanx-distal"),
    ("thumb-phalanx-distal", "thumb-tip"),

    ("wrist", "index-finger-metacarpal"),
    ("index-finger-metacarpal", "index-finger-phalanx-proximal"),
    ("index-finger-phalanx-proximal", "index-finger-phalanx-intermediate"),
    ("index-finger-phalanx-intermediate", "index-finger-phalanx-distal"),
    ("index-finger-phalanx-distal", "index-finger-tip"),

    ("wrist", "middle-finger-metacarpal"),
    ("middle-finger-metacarpal", "middle-finger-phalanx-proximal"),
    ("middle-finger-phalanx-proximal", "middle-finger-phalanx-intermediate"),
    ("middle-finger-phalanx-intermediate", "middle-finger-phalanx-distal"),
    ("middle-finger-phalanx-distal", "middle-finger-tip"),

    ("wrist", "ring-finger-metacarpal"),
    ("ring-finger-metacarpal", "ring-finger-phalanx-proximal"),
    ("ring-finger-phalanx-proximal", "ring-finger-phalanx-intermediate"),
    ("ring-finger-phalanx-intermediate", "ring-finger-phalanx-distal"),
    ("ring-finger-phalanx-distal", "ring-finger-tip"),

    ("wrist", "pinky-finger-metacarpal"),
    ("pinky-finger-metacarpal", "pinky-finger-phalanx-proximal"),
    ("pinky-finger-phalanx-proximal", "pinky-finger-phalanx-intermediate"),
    ("pinky-finger-phalanx-intermediate", "pinky-finger-phalanx-distal"),
    ("pinky-finger-phalanx-distal", "pinky-finger-tip"),
]


def draw_webxr_hand(draw, default_pos, quest_origin):
    if draw is None or quest_origin is None:
        return

    hand = get_hand_packet("right")
    if not hand:
        return

    points = []
    point_map = {}

    for name, joint in hand.items():
        if not isinstance(joint, dict) or "p" not in joint:
            continue

        p_isaac = webxr_pos_to_isaac(joint["p"])
        p_world = default_pos[0] + (p_isaac - quest_origin).to(default_pos.device) * 0.50
        tup = tuple(float(v) for v in p_world.detach().cpu().tolist())
        point_map[name] = tup
        points.append(tup)

    if points:
        draw.draw_points(points, [(1.0, 0.35, 0.0, 1.0)] * len(points), [8.0] * len(points))

    starts = []
    ends = []
    for a, b in HAND_BONES:
        if a in point_map and b in point_map:
            starts.append(point_map[a])
            ends.append(point_map[b])

    if starts:
        draw.draw_lines(starts, ends, [(1.0, 0.35, 0.0, 1.0)] * len(starts), [2.0] * len(starts))


def draw_axis_frame(draw, origin, quat_wxyz, scale=0.12, z_offset=0.0):
    if draw is None:
        return

    origin = origin.clone()
    origin[2] += z_offset

    rot = quat_wxyz_to_matrix(quat_wxyz.detach().cpu())
    o = tuple(float(v) for v in origin.detach().cpu().tolist())

    x_end = origin + rot[:, 0].to(origin.device) * scale
    y_end = origin + rot[:, 1].to(origin.device) * scale
    z_end = origin + rot[:, 2].to(origin.device) * scale

    draw.draw_lines(
        [o, o, o],
        [
            tuple(float(v) for v in x_end.detach().cpu().tolist()),
            tuple(float(v) for v in y_end.detach().cpu().tolist()),
            tuple(float(v) for v in z_end.detach().cpu().tolist()),
        ],
        [
            (1.0, 0.0, 0.0, 1.0),
            (0.0, 1.0, 0.0, 1.0),
            (0.0, 0.2, 1.0, 1.0),
        ],
        [4.0, 4.0, 4.0],
    )


# ------------------------------------------------------------
# Scene
# ------------------------------------------------------------

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=3000.0, color=(0.75, 0.75, 0.75)),
    )

    robot: ArticulationCfg = G1_INSPIRE_FTP_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot",
    )


sim_cfg = sim_utils.SimulationCfg(dt=1 / 120, device=args_cli.device)
sim = sim_utils.SimulationContext(sim_cfg)
sim.set_camera_view([2.5, -2.5, 1.6], [0.0, 0.0, 1.0])

scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.5))
sim.reset()

robot = scene["robot"]
draw = setup_debug_draw()

right_arm_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

right_joint_ids, right_joint_names_found = robot.find_joints(right_arm_joint_names)
right_body_ids, right_body_names_found = robot.find_bodies(["right_wrist_yaw_link"])
right_body_id = right_body_ids[0]

print("right_joint_ids:", right_joint_ids)
print("right_joint_names:", right_joint_names_found)
print("right_body_id:", right_body_id, right_body_names_found)

default_q = robot.data.default_joint_pos.clone()
zero_v = torch.zeros_like(default_q)

# Comfortable fixed arm pose so only wrist orientation changes.
base_q = default_q.clone()

name_to_local = {name: i for i, name in enumerate(right_joint_names_found)}

def set_local(q, name, value):
    jid = right_joint_ids[name_to_local[name]]
    q[:, jid] = value


set_local(base_q, "right_shoulder_pitch_joint", -0.20)
set_local(base_q, "right_shoulder_roll_joint", -0.35)
set_local(base_q, "right_shoulder_yaw_joint", 0.00)
set_local(base_q, "right_elbow_joint", 0.65)


# ------------------------------------------------------------
# Random pose generation
# ------------------------------------------------------------

# Conservative ranges first. These are enough for random orientation coverage.
ROLL_RANGE = (-1.40, 1.40)
PITCH_RANGE = (-1.20, 1.20)
YAW_RANGE = (-1.20, 1.20)

# Seed so if needed we can reproduce a sequence.
RANDOM_SEED = int(time.time())
random.seed(RANDOM_SEED)


def sample_random_pose(sample_idx):
    """
    Random robot wrist orientation target.

    Designed for better coverage than the 100-sample script:
      - wider joint ranges
      - more diagonal combinations
      - explicit palm-floor / palm-sky style poses
      - repeated neutral references
      - extreme-but-safe wrist rotations
    """

    # Regular neutral references help the learned model anchor the center.
    if sample_idx % 20 == 0:
        return {
            "mode": "neutral_reference",
            "roll": random.uniform(-0.12, 0.12),
            "pitch": random.uniform(-0.12, 0.12),
            "yaw": random.uniform(-0.12, 0.12),
        }

    r = random.random()

    # 0.00 - 0.18: mostly roll, because roll contributed a lot visually.
    if r < 0.18:
        return {
            "mode": "mostly_roll",
            "roll": random.uniform(*ROLL_RANGE),
            "pitch": random.uniform(-0.20, 0.20),
            "yaw": random.uniform(-0.20, 0.20),
        }

    # 0.18 - 0.36: mostly pitch.
    if r < 0.36:
        return {
            "mode": "mostly_pitch",
            "roll": random.uniform(-0.20, 0.20),
            "pitch": random.uniform(*PITCH_RANGE),
            "yaw": random.uniform(-0.20, 0.20),
        }

    # 0.36 - 0.50: mostly yaw.
    if r < 0.50:
        return {
            "mode": "mostly_yaw",
            "roll": random.uniform(-0.20, 0.20),
            "pitch": random.uniform(-0.20, 0.20),
            "yaw": random.uniform(*YAW_RANGE),
        }

    # 0.50 - 0.68: roll + pitch diagonal.
    # This is important for palm sky/floor, because floor/sky often was not one pure joint.
    if r < 0.68:
        sign = random.choice([-1.0, 1.0])
        return {
            "mode": "roll_pitch_diagonal",
            "roll": sign * random.uniform(0.55, 1.35),
            "pitch": sign * random.uniform(0.45, 1.15),
            "yaw": random.uniform(-0.25, 0.25),
        }

    # 0.68 - 0.80: opposite roll/pitch diagonal.
    if r < 0.80:
        sign = random.choice([-1.0, 1.0])
        return {
            "mode": "roll_pitch_opposite_diagonal",
            "roll": sign * random.uniform(0.55, 1.35),
            "pitch": -sign * random.uniform(0.45, 1.15),
            "yaw": random.uniform(-0.25, 0.25),
        }

    # 0.80 - 0.90: palm-floor / palm-sky biased extremes.
    # These are intentionally strong combinations because palm facing floor was weak.
    if r < 0.90:
        floor_sky_templates = [
            # strong roll/pitch combinations
            (1.25, 0.85, 0.00),
            (-1.25, -0.85, 0.00),
            (1.15, -0.85, 0.00),
            (-1.15, 0.85, 0.00),

            # add yaw variants
            (1.10, 0.75, 0.65),
            (1.10, 0.75, -0.65),
            (-1.10, -0.75, 0.65),
            (-1.10, -0.75, -0.65),

            # pitch-heavy variants
            (0.45, 1.15, 0.35),
            (-0.45, -1.15, -0.35),
            (0.45, -1.15, 0.35),
            (-0.45, 1.15, -0.35),
        ]

        base = random.choice(floor_sky_templates)
        return {
            "mode": "floor_sky_biased",
            "roll": base[0] + random.uniform(-0.15, 0.15),
            "pitch": base[1] + random.uniform(-0.15, 0.15),
            "yaw": base[2] + random.uniform(-0.15, 0.15),
        }

    # 0.90 - 1.00: fully random combined pose.
    return {
        "mode": "combined_random_wide",
        "roll": random.uniform(*ROLL_RANGE),
        "pitch": random.uniform(*PITCH_RANGE),
        "yaw": random.uniform(*YAW_RANGE),
    }


def apply_pose(pose):
    q = base_q.clone()

    set_local(q, "right_wrist_roll_joint", pose["roll"])
    set_local(q, "right_wrist_pitch_joint", pose["pitch"])
    set_local(q, "right_wrist_yaw_joint", pose["yaw"])

    robot.write_joint_state_to_sim(q, zero_v)
    robot.set_joint_position_target(q)
    return q


# ------------------------------------------------------------
# Dataset state
# ------------------------------------------------------------

samples = []
skipped = []
sample_idx = 0
pose = sample_random_pose(sample_idx)
current_q = apply_pose(pose)
quest_origin = None
step = 0
out_path = Path(args_cli.out)

print("")
print("================================================")
print("RIGHT HAND RANDOM ORIENTATION CALIBRATION - 300 WIDE COVERAGE")
print("================================================")
print(f"Target samples: {args_cli.num_samples}")
print(f"Output: {out_path}")
print(f"Random seed: {RANDOM_SEED}")
print("")
print("Robot shows a random right wrist/palm orientation.")
print("Move your real right hand until the WebXR hand orientation visually matches the robot palm orientation.")
print("Do NOT worry about matching wrist position in space.")
print("")
print("ENTER      save sample and advance")
print("s + ENTER  skip current pose")
print("w + ENTER  write/save JSON now, keep going")
print("q + ENTER  write/save JSON and quit")
print("================================================")
print("")
print(f"[POSE {sample_idx+1}/{args_cli.num_samples}] {pose}")


def save_json(final=False):
    payload = {
        "description": "Right hand random WebXR-to-G1 wrist orientation calibration. Match orientation only; position is not used.",
        "created_time": time.time(),
        "final": final,
        "target_num_samples": args_cli.num_samples,
        "num_saved_samples": len(samples),
        "num_skipped": len(skipped),
        "random_seed": RANDOM_SEED,
        "roll_range": ROLL_RANGE,
        "pitch_range": PITCH_RANGE,
        "yaw_range": YAW_RANGE,
        "samples": samples,
        "skipped": skipped,
    }

    out_path.write_text(json.dumps(payload, indent=2))
    print(f"[SAVE] wrote {out_path} with {len(samples)} samples, skipped={len(skipped)}, final={final}")


def capture_sample():
    wrist = get_joint("right", "wrist")
    if wrist is None:
        print("[WARN] no right WebXR wrist packet; sample not saved")
        return None

    palm_q, palm_normal, palm_axes = estimate_webxr_palm_frame("right")

    qx, qy, qz, qw = wrist.get("q", [0.0, 0.0, 0.0, 1.0])
    web_wrist_raw_wxyz = [qw, qx, qy, qz]
    web_wrist_isaac = webxr_quat_xyzw_to_isaac_wxyz(wrist["q"])

    robot_pos = robot.data.body_link_pos_w[0, right_body_id]
    robot_quat = robot.data.body_link_quat_w[0, right_body_id]

    sample = {
        "sample_index": len(samples),
        "shown_pose_index": sample_idx,
        "shown_pose_mode": pose["mode"],

        "robot_right_wrist_joint_values_roll_pitch_yaw": [
            pose["roll"],
            pose["pitch"],
            pose["yaw"],
        ],

        "robot_right_wrist_yaw_link_pos_w": robot_pos.detach().cpu().tolist(),
        "robot_right_wrist_yaw_link_quat_wxyz": robot_quat.detach().cpu().tolist(),

        "webxr_right_wrist_pos_raw_xyz": wrist["p"],
        "webxr_right_wrist_pos_isaac_xyz": webxr_pos_to_isaac(wrist["p"]).detach().cpu().tolist(),
        "webxr_right_wrist_quat_raw_xyzw": wrist.get("q", None),
        "webxr_right_wrist_quat_raw_wxyz": web_wrist_raw_wxyz,
        "webxr_right_wrist_quat_isaac_wxyz": web_wrist_isaac.detach().cpu().tolist(),

        "webxr_right_palm_quat_isaac_wxyz": None if palm_q is None else palm_q.detach().cpu().tolist(),
        "webxr_right_palm_normal_isaac_xyz": None if palm_normal is None else palm_normal.detach().cpu().tolist(),
        "webxr_right_palm_axes_isaac": palm_axes,

        "webxr_right_keypoints": collect_webxr_keypoints("right"),

        "packet_count": latest["count"],
        "packet_age_sec": time.time() - latest["time"] if latest["time"] > 0 else None,
        "saved_time": time.time(),
    }

    return sample


while simulation_app.is_running():
    step += 1

    scene.update(sim.get_physics_dt())

    # Keep programmed pose fixed.
    robot.set_joint_position_target(current_q)
    scene.write_data_to_sim()
    sim.step()

    wrist = get_joint("right", "wrist")
    if wrist is not None and quest_origin is None:
        quest_origin = webxr_pos_to_isaac(wrist["p"])
        print("[INIT] right Quest origin:", quest_origin.tolist())

    # Draw WebXR hand and frames.
    if step % 2 == 0:
        safe_clear_draw(draw)

        if quest_origin is not None:
            draw_webxr_hand(draw, robot.data.body_link_pos_w[:, right_body_id], quest_origin)

        robot_pos = robot.data.body_link_pos_w[0, right_body_id]
        robot_quat = robot.data.body_link_quat_w[0, right_body_id]

        # Robot target/actual wrist frame
        draw_axis_frame(draw, robot_pos, robot_quat, scale=0.12, z_offset=0.0)

        # WebXR palm frame floating above robot wrist
        palm_q, _, _ = estimate_webxr_palm_frame("right")
        if palm_q is not None:
            draw_axis_frame(draw, robot_pos, palm_q, scale=0.14, z_offset=0.20)

    if input_events:
        cmd = input_events.pop(0)

        if cmd == "w":
            save_json(final=False)
            continue

        if cmd == "q":
            save_json(final=True)
            print("[QUIT] saved and quitting")
            break

        if cmd == "s":
            skipped.append({
                "shown_pose_index": sample_idx,
                "pose": pose,
                "time": time.time(),
                "reason": "manual_skip",
            })
            print(f"[SKIP] pose {sample_idx+1}: {pose}")

            sample_idx += 1
            pose = sample_random_pose(sample_idx)
            current_q = apply_pose(pose)
            print(f"[POSE {sample_idx+1}/{args_cli.num_samples}] saved={len(samples)} skipped={len(skipped)} {pose}")
            continue

        # ENTER = save sample and advance
        sample = capture_sample()
        if sample is not None:
            samples.append(sample)
            print(f"[SAVED] {len(samples)}/{args_cli.num_samples} pose={pose}")

            # Save after every sample so you don't lose work.
            save_json(final=False)

        if len(samples) >= args_cli.num_samples:
            save_json(final=True)
            print("[DONE] reached target sample count")
            break

        sample_idx += 1
        pose = sample_random_pose(sample_idx)
        current_q = apply_pose(pose)
        print(f"[POSE {sample_idx+1}/{args_cli.num_samples}] saved={len(samples)} skipped={len(skipped)} {pose}")

simulation_app.close()
