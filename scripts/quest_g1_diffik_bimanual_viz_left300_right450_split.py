import argparse
import json
import math
import socket
import threading
import time

import torch
import torch.nn as nn

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.utils import configclass
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG


# ------------------------------------------------------------
# UDP WebXR stream
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

            if latest["count"] % 120 == 0:
                hands = list(obj.get("hands", {}).keys())
                print("[UDP]", latest["count"], hands)
        except Exception as e:
            print("[UDP ERROR]", e)


threading.Thread(target=udp_server, daemon=True).start()


def get_joint(hand_name: str, joint_name: str):
    pkt = latest["packet"]
    if not pkt:
        return None

    hand = pkt.get("hands", {}).get(hand_name)
    if not hand:
        return None

    j = hand.get(joint_name)
    if isinstance(j, dict) and "p" in j:
        return j

    return None


def get_wrist(hand_name: str):
    return get_joint(hand_name, "wrist")


# ------------------------------------------------------------
# Coordinate / quaternion helpers
# ------------------------------------------------------------

def webxr_pos_to_isaac(p):
    """
    Working mapping from previous tests:
    WebXR [x, y, z] -> Isaac [-z, -x, y]
    """
    x, y, z = p
    return torch.tensor([-z, -x, y], dtype=torch.float32)


def quat_normalize(q):
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1e-8)


def quat_inv(q):
    # q is wxyz
    out = q.clone()
    out[..., 1:] *= -1.0
    return quat_normalize(out)


def quat_mul(q1, q2):
    # q1, q2 are wxyz
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


def quat_slerp(q0, q1, t: float):
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)

    dot = torch.sum(q0 * q1, dim=-1, keepdim=True)

    # take shortest path
    q1 = torch.where(dot < 0.0, -q1, q1)
    dot = torch.abs(dot).clamp(-1.0, 1.0)

    if torch.all(dot > 0.9995):
        return quat_normalize((1.0 - t) * q0 + t * q1)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * t
    sin_theta = torch.sin(theta)

    s0 = torch.cos(theta) - dot * sin_theta / sin_theta_0
    s1 = sin_theta / sin_theta_0

    return quat_normalize(s0 * q0 + s1 * q1)


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
    # Robust enough for our single-matrix wrist orientation use.
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


def webxr_quat_xyzw_to_isaac_wxyz(q_xyzw, device):
    """
    WebXR gives [qx, qy, qz, qw].
    Isaac uses [qw, qx, qy, qz].

    We also transform quaternion basis using the same coordinate transform
    as position:
        web [x,y,z] -> isaac [-z,-x,y]
    """
    qx, qy, qz, qw = q_xyzw
    q_web = torch.tensor([qw, qx, qy, qz], dtype=torch.float32)

    r_web = quat_wxyz_to_matrix(q_web)

    # Mapping matrix A: isaac = A @ web
    a = torch.tensor(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=torch.float32,
    )

    r_isaac = a @ r_web @ a.T
    return matrix_to_quat_wxyz(r_isaac).to(device).unsqueeze(0)


def make_target_quat_from_webxr(hand_name, default_quat, device, quat_origin_store, wrist_joint, orient_blend):
    if wrist_joint is None or "q" not in wrist_joint:
        return default_quat.clone()

    q_now = webxr_quat_xyzw_to_isaac_wxyz(wrist_joint["q"], device)

    if quat_origin_store[hand_name] is None:
        quat_origin_store[hand_name] = q_now.clone()
        return default_quat.clone()

    q_delta = quat_mul(q_now, quat_inv(quat_origin_store[hand_name]))

    # World-relative delta applied to robot default orientation.
    q_target = quat_mul(q_delta, default_quat)

    # Blend in orientation slowly for safety.
    return quat_slerp(default_quat, q_target, orient_blend)


# ------------------------------------------------------------
# WebXR hand skeleton visualization
# ------------------------------------------------------------

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


def viz_hand(draw, hand_name, default_pos, quest_origin, color):
    if draw is None:
        return

    pkt = latest["packet"]
    if not pkt:
        return

    hand = pkt.get("hands", {}).get(hand_name)
    if not hand or quest_origin is None:
        return

    points = []
    point_map = {}

    for name, joint in hand.items():
        if not isinstance(joint, dict) or "p" not in joint:
            continue

        p_isaac = webxr_pos_to_isaac(joint["p"])
        delta = (p_isaac - quest_origin).to(default_pos.device)
        p_world = default_pos[0] + delta * VIZ_SCALE
        tup = tuple(float(v) for v in p_world.detach().cpu().tolist())

        point_map[name] = tup
        points.append(tup)

    if points:
        draw.draw_points(points, [color] * len(points), [8.0] * len(points))

    line_starts = []
    line_ends = []

    for a, b in HAND_BONES:
        if a in point_map and b in point_map:
            line_starts.append(point_map[a])
            line_ends.append(point_map[b])

    if line_starts:
        draw.draw_lines(line_starts, line_ends, [color] * len(line_starts), [2.0] * len(line_starts))



# ------------------------------------------------------------
# G1 Inspire finger retargeting
# ------------------------------------------------------------

def _joint_pos(hand_name, joint_name, device):
    j = get_joint(hand_name, joint_name)
    if j is None or "p" not in j:
        return None
    return webxr_pos_to_isaac(j["p"]).to(device)


def _angle_between(v1, v2):
    v1 = v1 / torch.clamp(torch.linalg.norm(v1), min=1e-6)
    v2 = v2 / torch.clamp(torch.linalg.norm(v2), min=1e-6)
    dot = torch.clamp(torch.dot(v1, v2), -1.0, 1.0)
    return torch.acos(dot)


def _finger_curl(hand_name, finger, device):
    """
    Returns a rough curl value in [0, 1].
    Uses angle between metacarpal->proximal and proximal->tip.
    """
    if finger == "thumb":
        a = _joint_pos(hand_name, "thumb-metacarpal", device)
        b = _joint_pos(hand_name, "thumb-phalanx-proximal", device)
        c = _joint_pos(hand_name, "thumb-tip", device)
    else:
        a = _joint_pos(hand_name, f"{finger}-finger-metacarpal", device)
        b = _joint_pos(hand_name, f"{finger}-finger-phalanx-proximal", device)
        c = _joint_pos(hand_name, f"{finger}-finger-tip", device)

    if a is None or b is None or c is None:
        return None

    v1 = b - a
    v2 = c - b

    angle = _angle_between(v1, v2)

    # Rough range:
    # open finger: small angle
    # curled finger: larger angle
    curl = torch.clamp((angle - 0.25) / 1.35, 0.0, 1.0)
    return float(curl.detach().cpu())


def _thumb_yaw(hand_name, device):
    """
    Very rough thumb yaw/open-spread estimate.
    Uses thumb direction relative to index direction.
    """
    wrist = _joint_pos(hand_name, "wrist", device)
    thumb = _joint_pos(hand_name, "thumb-tip", device)
    index = _joint_pos(hand_name, "index-finger-metacarpal", device)

    if wrist is None or thumb is None or index is None:
        return 0.0

    vt = thumb - wrist
    vi = index - wrist
    angle = _angle_between(vt, vi)
    spread = torch.clamp((angle - 0.35) / 1.0, 0.0, 1.0)
    return float(spread.detach().cpu())


def compute_g1_finger_targets(hand_name, device):
    """
    G1 Inspire hand has 12 joints per hand.

    Order for left:
      L_index_proximal
      L_middle_proximal
      L_pinky_proximal
      L_ring_proximal
      L_thumb_proximal_yaw
      L_index_intermediate
      L_middle_intermediate
      L_pinky_intermediate
      L_ring_intermediate
      L_thumb_proximal_pitch
      L_thumb_intermediate
      L_thumb_distal

    Same for right with R_ prefix.

    Returns tensor shape [1, 12].
    """
    index = _finger_curl(hand_name, "index", device)
    middle = _finger_curl(hand_name, "middle", device)
    ring = _finger_curl(hand_name, "ring", device)
    pinky = _finger_curl(hand_name, "pinky", device)
    thumb = _finger_curl(hand_name, "thumb", device)

    if index is None and middle is None and ring is None and pinky is None and thumb is None:
        return None

    # Fill missing values with 0/open.
    index = 0.0 if index is None else index
    middle = 0.0 if middle is None else middle
    ring = 0.0 if ring is None else ring
    pinky = 0.0 if pinky is None else pinky
    thumb = 0.0 if thumb is None else thumb

    thumb_yaw = _thumb_yaw(hand_name, device)

    # Tunables.
    # If fingers curl wrong direction, flip FINGER_SIGN.
    FINGER_SIGN = 1.0

    # Joint ranges are conservative first.
    prox_scale = 0.85
    inter_scale = 0.95
    thumb_pitch_scale = 0.75
    thumb_yaw_scale = 0.55

    vals = [
        FINGER_SIGN * prox_scale * index,
        FINGER_SIGN * prox_scale * middle,
        FINGER_SIGN * prox_scale * pinky,
        FINGER_SIGN * prox_scale * ring,
        FINGER_SIGN * thumb_yaw_scale * thumb_yaw,

        FINGER_SIGN * inter_scale * index,
        FINGER_SIGN * inter_scale * middle,
        FINGER_SIGN * inter_scale * pinky,
        FINGER_SIGN * inter_scale * ring,
        FINGER_SIGN * thumb_pitch_scale * thumb,

        FINGER_SIGN * inter_scale * thumb,
        FINGER_SIGN * inter_scale * thumb,
    ]

    return torch.tensor([vals], dtype=torch.float32, device=device)




# ------------------------------------------------------------
# Learned right wrist mapping
# ------------------------------------------------------------

LEARNED_WRIST_BLEND = 0.90
RIGHT_WRIST_MODEL_PATH = "/home/amro/IsaacLab_5/right_wrist_mapping_model_450_targeted.pt"
LEFT_WRIST_MODEL_PATH = "/home/amro/IsaacLab_5/left_wrist_mapping_model_300.pt"


class WristMapMLP(nn.Module):
    def __init__(self, in_dim=41, out_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 128),
            nn.LayerNorm(128),
            nn.GELU(),

            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.GELU(),

            nn.Linear(64, out_dim),
        )

    def forward(self, x):
        return self.net(x)


def load_learned_wrist_model(device):
    global learned_wrist_model, learned_wrist_x_mean, learned_wrist_x_std

    ckpt = torch.load(RIGHT_WRIST_MODEL_PATH, map_location="cpu")

    model = WristMapMLP(
        in_dim=ckpt["input_dim"],
        out_dim=3,
    ).to(device)

    model.load_state_dict(ckpt["model"])
    model.eval()

    learned_wrist_model = model
    learned_wrist_x_mean = ckpt["x_mean"].to(device)
    learned_wrist_x_std = ckpt["x_std"].to(device)

    print("[LEARNED WRIST] loaded:", RIGHT_WRIST_MODEL_PATH)
    print("[LEARNED WRIST] blend:", LEARNED_WRIST_BLEND)


def estimate_learned_palm_frame(hand_name, device):
    wrist = get_joint(hand_name, "wrist")
    index = get_joint(hand_name, "index-finger-metacarpal")
    middle = get_joint(hand_name, "middle-finger-metacarpal")
    pinky = get_joint(hand_name, "pinky-finger-metacarpal")

    if wrist is None or index is None or middle is None or pinky is None:
        return None, None

    pw = webxr_pos_to_isaac(wrist["p"]).to(device)
    pi = webxr_pos_to_isaac(index["p"]).to(device)
    pm = webxr_pos_to_isaac(middle["p"]).to(device)
    pp = webxr_pos_to_isaac(pinky["p"]).to(device)

    across = pi - pp
    across = across / torch.clamp(torch.linalg.norm(across), min=1e-6)

    forward = pm - pw
    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-6)

    normal = torch.cross(across, forward, dim=0)
    normal = normal / torch.clamp(torch.linalg.norm(normal), min=1e-6)

    forward = torch.cross(normal, across, dim=0)
    forward = forward / torch.clamp(torch.linalg.norm(forward), min=1e-6)

    r = torch.stack([across, forward, normal], dim=1).detach().cpu()
    palm_q = matrix_to_quat_wxyz(r).to(device)

    return palm_q, normal


def build_learned_right_wrist_feature(device):
    """
    Same features as 400-big training:
      wrist quat: 4
      palm quat: 4
      palm normal: 3
      wrist->5 fingertip unit vectors: 15
      wrist->5 metacarpal unit vectors: 15
    total: 41
    """
    wrist = get_joint("right", "wrist")
    if wrist is None or "q" not in wrist:
        return None

    wrist_q = webxr_quat_xyzw_to_isaac_wxyz(wrist["q"], device).squeeze(0)
    wrist_q = wrist_q / torch.clamp(torch.linalg.norm(wrist_q), min=1e-8)

    palm_q, palm_n = estimate_learned_palm_frame("right", device)
    if palm_q is None or palm_n is None:
        return None

    palm_q = palm_q / torch.clamp(torch.linalg.norm(palm_q), min=1e-8)

    joint_names = [
        "wrist",

        "thumb-tip",
        "index-finger-tip",
        "middle-finger-tip",
        "ring-finger-tip",
        "pinky-finger-tip",

        "thumb-metacarpal",
        "index-finger-metacarpal",
        "middle-finger-metacarpal",
        "ring-finger-metacarpal",
        "pinky-finger-metacarpal",
    ]

    kps = {}
    for name in joint_names:
        j = get_joint("right", name)
        if j is None or "p" not in j:
            return None
        kps[name] = webxr_pos_to_isaac(j["p"]).to(device)

    wrist_p = kps["wrist"]

    feats = [wrist_q, palm_q, palm_n]

    for name in [
        "thumb-tip",
        "index-finger-tip",
        "middle-finger-tip",
        "ring-finger-tip",
        "pinky-finger-tip",
    ]:
        v = kps[name] - wrist_p
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-6)
        feats.append(v)

    for name in [
        "thumb-metacarpal",
        "index-finger-metacarpal",
        "middle-finger-metacarpal",
        "ring-finger-metacarpal",
        "pinky-finger-metacarpal",
    ]:
        v = kps[name] - wrist_p
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-6)
        feats.append(v)

    x = torch.cat(feats, dim=0).unsqueeze(0)
    return x




# Load left learned wrist model using the same MLP architecture.
_left_ckpt = torch.load(LEFT_WRIST_MODEL_PATH, map_location="cpu")
left_learned_wrist_model = WristMapMLP(
    in_dim=_left_ckpt.get("input_dim", 41),
    out_dim=3,
).to(args_cli.device)
left_learned_wrist_model.load_state_dict(_left_ckpt["model"])
left_learned_wrist_model.eval()
left_x_mean = _left_ckpt["x_mean"].to(args_cli.device)
left_x_std = _left_ckpt["x_std"].to(args_cli.device)
print("[LEFT LEARNED WRIST] loaded:", LEFT_WRIST_MODEL_PATH)

def build_learned_left_wrist_feature(device):
    """
    Same features as left training:
      wrist quat: 4
      palm quat: 4
      palm normal: 3
      wrist->5 fingertip unit vectors: 15
      wrist->5 metacarpal unit vectors: 15
    total: 41
    """
    wrist = get_joint("left", "wrist")
    if wrist is None or "q" not in wrist:
        return None

    wrist_q = webxr_quat_xyzw_to_isaac_wxyz(wrist["q"], device).squeeze(0)
    wrist_q = wrist_q / torch.clamp(torch.linalg.norm(wrist_q), min=1e-8)

    palm_q, palm_n = estimate_learned_palm_frame("left", device)
    if palm_q is None or palm_n is None:
        return None

    palm_q = palm_q / torch.clamp(torch.linalg.norm(palm_q), min=1e-8)

    joint_names = [
        "wrist",

        "thumb-tip",
        "index-finger-tip",
        "middle-finger-tip",
        "ring-finger-tip",
        "pinky-finger-tip",

        "thumb-metacarpal",
        "index-finger-metacarpal",
        "middle-finger-metacarpal",
        "ring-finger-metacarpal",
        "pinky-finger-metacarpal",
    ]

    kps = {}
    for name in joint_names:
        j = get_joint("left", name)
        if j is None or "p" not in j:
            return None
        kps[name] = webxr_pos_to_isaac(j["p"]).to(device)

    wrist_p = kps["wrist"]

    feats = [wrist_q, palm_q, palm_n]

    for name in [
        "thumb-tip",
        "index-finger-tip",
        "middle-finger-tip",
        "ring-finger-tip",
        "pinky-finger-tip",
    ]:
        v = kps[name] - wrist_p
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-6)
        feats.append(v)

    for name in [
        "thumb-metacarpal",
        "index-finger-metacarpal",
        "middle-finger-metacarpal",
        "ring-finger-metacarpal",
        "pinky-finger-metacarpal",
    ]:
        v = kps[name] - wrist_p
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-6)
        feats.append(v)

    x = torch.cat(feats, dim=0).unsqueeze(0)
    return x


def predict_left_wrist_roll_pitch_yaw(device):
    x = build_learned_left_wrist_feature(device)
    if x is None:
        return None

    with torch.no_grad():
        x = (x - left_x_mean) / left_x_std
        pred = left_learned_wrist_model(x)

    return pred


def predict_right_wrist_roll_pitch_yaw(device):
    if learned_wrist_model is None:
        return None

    x = build_learned_right_wrist_feature(device)
    if x is None:
        return None

    x = (x - learned_wrist_x_mean) / learned_wrist_x_std

    with torch.no_grad():
        pred = learned_wrist_model(x)

    return pred  # shape [1,3] roll,pitch,yaw


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
sim.set_camera_view([2.5, 2.5, 1.8], [0.0, 0.0, 1.0])

scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.5))
sim.reset()

robot = scene["robot"]
draw = setup_debug_draw()


# ------------------------------------------------------------
# Arm setup
# ------------------------------------------------------------

left_joint_names = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_yaw_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
]

right_joint_names = [
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
]

left_joint_ids, left_joint_names_found = robot.find_joints(left_joint_names)
right_joint_ids, right_joint_names_found = robot.find_joints(right_joint_names)

left_body_ids, left_body_names_found = robot.find_bodies(["left_wrist_yaw_link"])
right_body_ids, right_body_names_found = robot.find_bodies(["right_wrist_yaw_link"])

left_body_id = left_body_ids[0]
right_body_id = right_body_ids[0]

# Working values from your tests.
LEFT_JACOBIAN_INDEX = 27
RIGHT_JACOBIAN_INDEX = 28

print("left_joint_ids:", left_joint_ids)
print("left_joint_names:", left_joint_names_found)
print("left_body_id:", left_body_id, left_body_names_found)
print("using LEFT_JACOBIAN_INDEX:", LEFT_JACOBIAN_INDEX)

print("right_joint_ids:", right_joint_ids)
print("right_joint_names:", right_joint_names_found)

# Split architecture: right wrist orientation joints are controlled separately.
right_wrist_orient_joint_names = [
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

right_wrist_orient_joint_ids, right_wrist_orient_names = robot.find_joints(right_wrist_orient_joint_names)

left_wrist_orient_joint_names = [
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
]

left_wrist_orient_joint_ids, left_wrist_orient_names = robot.find_joints(left_wrist_orient_joint_names)

print("right_wrist_orient_joint_ids:", right_wrist_orient_joint_ids)
print("right_wrist_orient_names:", right_wrist_orient_names)
print("left_wrist_orient_joint_ids:", left_wrist_orient_joint_ids)
print("left_wrist_orient_names:", left_wrist_orient_names)

print("right_body_id:", right_body_id, right_body_names_found)
print("using RIGHT_JACOBIAN_INDEX:", RIGHT_JACOBIAN_INDEX)

# Finger joints exposed by G1 Inspire.
left_finger_joint_names = [
    "L_index_proximal_joint",
    "L_middle_proximal_joint",
    "L_pinky_proximal_joint",
    "L_ring_proximal_joint",
    "L_thumb_proximal_yaw_joint",
    "L_index_intermediate_joint",
    "L_middle_intermediate_joint",
    "L_pinky_intermediate_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
]

right_finger_joint_names = [
    "R_index_proximal_joint",
    "R_middle_proximal_joint",
    "R_pinky_proximal_joint",
    "R_ring_proximal_joint",
    "R_thumb_proximal_yaw_joint",
    "R_index_intermediate_joint",
    "R_middle_intermediate_joint",
    "R_pinky_intermediate_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]

left_finger_joint_ids, left_finger_names_found = robot.find_joints(left_finger_joint_names)
right_finger_joint_ids, right_finger_names_found = robot.find_joints(right_finger_joint_names)

print("left_finger_joint_ids:", left_finger_joint_ids)
print("left_finger_names:", left_finger_names_found)
print("right_finger_joint_ids:", right_finger_joint_ids)
print("right_finger_names:", right_finger_names_found)



ik_cfg = DifferentialIKControllerCfg(
    command_type="position",
    use_relative_mode=False,
    ik_method="dls",
)

left_ik = DifferentialIKController(ik_cfg, num_envs=1, device=args_cli.device)
right_ik = DifferentialIKController(ik_cfg, num_envs=1, device=args_cli.device)

default_joint_pos = robot.data.default_joint_pos.clone()
zero_joint_vel = torch.zeros_like(default_joint_pos)

robot.write_joint_state_to_sim(default_joint_pos, zero_joint_vel)
robot.set_joint_position_target(default_joint_pos)
scene.write_data_to_sim()

left_ik.reset()
right_ik.reset()

sim.step()
scene.update(sim.get_physics_dt())

left_default_pos = robot.data.body_link_pos_w[:, left_body_id].clone()
left_default_quat = robot.data.body_link_quat_w[:, left_body_id].clone()

right_default_pos = robot.data.body_link_pos_w[:, right_body_id].clone()
right_default_quat = robot.data.body_link_quat_w[:, right_body_id].clone()

print("left_default_pos:", left_default_pos[0].detach().cpu().tolist())
print("left_default_quat:", left_default_quat[0].detach().cpu().tolist())
print("right_default_pos:", right_default_pos[0].detach().cpu().tolist())
print("right_default_quat:", right_default_quat[0].detach().cpu().tolist())

load_learned_wrist_model(args_cli.device)



# ------------------------------------------------------------
# Tunables
# ------------------------------------------------------------

DELTA_SCALE = 0.50
MAX_DELTA_M = 0.18

# Start modest. Increase later if orientation looks good.
ORIENT_BLEND = 0.60

POSTURE_GAIN = 0.08

LEFT_NATURAL_Q = torch.tensor(
    [-0.15, 0.22, 0.00, 0.35, 0.00, 0.00, 0.00],
    dtype=torch.float32,
    device=args_cli.device,
).unsqueeze(0)

RIGHT_NATURAL_Q = torch.tensor(
    [-0.15, -0.22, 0.00, 0.35],
    dtype=torch.float32,
    device=args_cli.device,
).unsqueeze(0)

Q_STEP_LIMITS = torch.tensor(
    [0.040, 0.040, 0.040, 0.060, 0.120, 0.120, 0.120],
    dtype=torch.float32,
    device=args_cli.device,
).unsqueeze(0)

PACKET_TIMEOUT = 1.0
VIZ_SCALE = 0.50

quest_origin = {
    "left": None,
    "right": None,
}

quat_origin = {
    "left": None,
    "right": None,
}

step = 0

print("READY.")
print("Bimanual G1 Inspire DiffIK + WebXR hand skeleton viz.")
print("BIMANUAL learned wrist mapping: LEFT 300 + RIGHT 450 split architecture enabled")
print("Start WebXR server + Quest. First detected wrist becomes origin for each hand.")
print("If orientation twists too much, set ORIENT_BLEND = 0.0 first.")


def control_arm(
    hand_name,
    joint_ids,
    body_id,
    jacobian_index,
    ik,
    default_pos,
    default_quat,
    natural_q,
):
    wrist = get_wrist(hand_name)
    packet_age = time.time() - latest["time"] if latest["time"] > 0 else 999.0
    has_fresh = wrist is not None and packet_age < PACKET_TIMEOUT

    if not has_fresh:
        q_hold = robot.data.joint_pos[:, joint_ids].clone()
        robot.set_joint_position_target(q_hold, joint_ids=joint_ids)
        return False, None, None, packet_age

    pos_now_cpu = webxr_pos_to_isaac(wrist["p"])

    if quest_origin[hand_name] is None:
        quest_origin[hand_name] = pos_now_cpu.clone()
        print(f"[INIT] {hand_name} Quest origin:", quest_origin[hand_name].tolist())

    delta_cpu = pos_now_cpu - quest_origin[hand_name]
    delta_cpu = torch.clamp(delta_cpu * DELTA_SCALE, -MAX_DELTA_M, MAX_DELTA_M)

    target_pos = default_pos.clone()
    target_pos[0] = default_pos[0] + delta_cpu.to(target_pos.device)

    target_quat = make_target_quat_from_webxr(
        hand_name=hand_name,
        default_quat=default_quat,
        device=args_cli.device,
        quat_origin_store=quat_origin,
        wrist_joint=wrist,
        orient_blend=ORIENT_BLEND,
    )

    # Position-only IK.
    # Orientation is handled separately by learned wrist joints.
    cmd = target_pos
    ee_quat_for_position_cmd = robot.data.body_link_quat_w[:, body_id]
    ik.set_command(cmd, ee_quat=ee_quat_for_position_cmd)

    ee_pos = robot.data.body_link_pos_w[:, body_id]
    ee_quat = robot.data.body_link_quat_w[:, body_id]

    jac_all = robot.root_physx_view.get_jacobians()
    jac = jac_all[:, jacobian_index, :, joint_ids]

    q = robot.data.joint_pos[:, joint_ids]
    q_des = ik.compute(ee_pos, ee_quat, jac, q)

    # Adaptive posture:
    # close to neutral = more elbow posture
    # farther reach = less posture bias so elbow can extend/retract
    reach = torch.linalg.norm(delta_cpu).item()
    adaptive_posture_gain = max(0.02, POSTURE_GAIN * (1.0 - min(reach / 0.18, 1.0)))

    q_des = (1.0 - adaptive_posture_gain) * q_des + adaptive_posture_gain * natural_q

    # Learned wrist is applied separately after arm IK.
    q_cur = robot.data.joint_pos[:, joint_ids]

    # Left still has 7 joints, right split IK has 4 joints.
    q_step_limits = Q_STEP_LIMITS[:, : q_des.shape[1]]

    q_des = torch.max(torch.min(q_des, q_cur + q_step_limits), q_cur - q_step_limits)

    robot.set_joint_position_target(q_des, joint_ids=joint_ids)

    return True, target_pos, delta_cpu, packet_age



# ------------------------------------------------------------
# Split learned wrist control
# ------------------------------------------------------------

SPLIT_LEARNED_WRIST_BLEND = 1.00
SPLIT_WRIST_MAX_STEP = 0.120


def apply_split_learned_left_wrist():
    """
    Apply learned left wrist roll/pitch/yaw directly to left wrist joints.
    """
    learned = predict_left_wrist_roll_pitch_yaw(args_cli.device)
    if learned is None:
        return

    q_cur = robot.data.joint_pos[:, left_wrist_orient_joint_ids]

    q_target = (
        (1.0 - SPLIT_LEARNED_WRIST_BLEND) * q_cur
        + SPLIT_LEARNED_WRIST_BLEND * learned
    )

    q_target = torch.max(
        torch.min(q_target, q_cur + SPLIT_WRIST_MAX_STEP),
        q_cur - SPLIT_WRIST_MAX_STEP,
    )

    robot.set_joint_position_target(q_target, joint_ids=left_wrist_orient_joint_ids)

    if step % 120 == 0:
        print(
            "[SPLIT LEARNED LEFT WRIST]",
            "pred=", learned.detach().cpu().tolist(),
            "q_cur=", q_cur.detach().cpu().tolist(),
            "q_target=", q_target.detach().cpu().tolist(),
        )


def apply_split_learned_right_wrist():
    """
    Apply learned right wrist roll/pitch/yaw directly to wrist joints.

    Arm IK owns shoulder/elbow position.
    Learned model owns wrist orientation.
    """
    learned = predict_right_wrist_roll_pitch_yaw(args_cli.device)
    if learned is None:
        return

    q_cur = robot.data.joint_pos[:, right_wrist_orient_joint_ids]

    q_target = (
        (1.0 - SPLIT_LEARNED_WRIST_BLEND) * q_cur
        + SPLIT_LEARNED_WRIST_BLEND * learned
    )

    q_target = torch.max(
        torch.min(q_target, q_cur + SPLIT_WRIST_MAX_STEP),
        q_cur - SPLIT_WRIST_MAX_STEP,
    )

    robot.set_joint_position_target(q_target, joint_ids=right_wrist_orient_joint_ids)

    if step % 120 == 0:
        print(
            "[SPLIT LEARNED WRIST]",
            "pred=", learned.detach().cpu().tolist(),
            "q_cur=", q_cur.detach().cpu().tolist(),
            "q_target=", q_target.detach().cpu().tolist(),
        )



while simulation_app.is_running():
    step += 1

    scene.update(sim.get_physics_dt())

    left_ok, left_target, left_delta, left_age = control_arm(
        hand_name="left",
        joint_ids=left_joint_ids,
        body_id=left_body_id,
        jacobian_index=LEFT_JACOBIAN_INDEX,
        ik=left_ik,
        default_pos=left_default_pos,
        default_quat=left_default_quat,
        natural_q=LEFT_NATURAL_Q,
    )

    # Split architecture: apply learned left wrist orientation after arm position IK.
    apply_split_learned_left_wrist()

    right_ok, right_target, right_delta, right_age = control_arm(
        hand_name="right",
        joint_ids=right_joint_ids,
        body_id=right_body_id,
        jacobian_index=RIGHT_JACOBIAN_INDEX,
        ik=right_ik,
        default_pos=right_default_pos,
        default_quat=right_default_quat,
        natural_q=RIGHT_NATURAL_Q,
    )

    # Split architecture: apply learned wrist orientation after arm position IK.
    apply_split_learned_right_wrist()

    # Draw incoming WebXR hand skeletons near the robot.
    if step % 2 == 0:
        safe_clear_draw(draw)
        viz_hand(draw, "left", left_default_pos, quest_origin["left"], (0.0, 0.7, 1.0, 1.0))
        viz_hand(draw, "right", right_default_pos, quest_origin["right"], (1.0, 0.35, 0.0, 1.0))


    # Finger retargeting: direct joint targets from WebXR curl.
    left_fingers = compute_g1_finger_targets("left", args_cli.device)
    if left_fingers is not None:
        robot.set_joint_position_target(left_fingers, joint_ids=left_finger_joint_ids)

    right_fingers = compute_g1_finger_targets("right", args_cli.device)
    if right_fingers is not None:
        robot.set_joint_position_target(right_fingers, joint_ids=right_finger_joint_ids)

    scene.write_data_to_sim()
    sim.step()

    if step % 120 == 0:
        left_ee = robot.data.body_link_pos_w[0, left_body_id].detach().cpu().tolist()
        right_ee = robot.data.body_link_pos_w[0, right_body_id].detach().cpu().tolist()

        print(
            f"[DEBUG] step={step} packets={latest['count']} "
            f"L_ok={left_ok} R_ok={right_ok} "
            f"L_ee={left_ee} R_ee={right_ee}"
        )

        if left_ok and left_delta is not None:
            print("  L_delta:", left_delta.tolist(), "L_target:", left_target[0].detach().cpu().tolist())
        if right_ok and right_delta is not None:
            print("  R_delta:", right_delta.tolist(), "R_target:", right_target[0].detach().cpu().tolist())


simulation_app.close()
