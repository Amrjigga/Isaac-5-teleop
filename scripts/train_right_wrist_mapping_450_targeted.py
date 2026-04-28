import json
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim


DATA_PATHS = [
    Path("~/IsaacLab_5/g1_webxr_right_random_calibration_100.json").expanduser(),
    Path("~/IsaacLab_5/g1_webxr_right_random_calibration_300.json").expanduser(),
    Path("~/IsaacLab_5/g1_webxr_right_targeted_weakness_calibration_50.json").expanduser(),
]

OUT_PATH = Path("~/IsaacLab_5/right_wrist_mapping_model_450_targeted.pt").expanduser()

torch.manual_seed(42)


def q_normalize(q):
    q = torch.tensor(q, dtype=torch.float32)
    return q / torch.clamp(torch.linalg.norm(q), min=1e-8)


def build_feature(sample):
    """
    Input features:
      - WebXR wrist quat in Isaac frame: 4
      - WebXR palm quat in Isaac frame: 4
      - WebXR palm normal: 3
      - wrist -> 5 fingertip unit vectors: 15
      - wrist -> 5 MCP/metacarpal unit vectors: 15

    Total: 41 features
    """
    feats = []

    wrist_q = q_normalize(sample["webxr_right_wrist_quat_isaac_wxyz"])
    palm_q = q_normalize(sample["webxr_right_palm_quat_isaac_wxyz"])
    palm_n = torch.tensor(sample["webxr_right_palm_normal_isaac_xyz"], dtype=torch.float32)

    feats.append(wrist_q)
    feats.append(palm_q)
    feats.append(palm_n)

    kps = sample["webxr_right_keypoints"]
    wrist_p = torch.tensor(kps["wrist"]["p_isaac_xyz"], dtype=torch.float32)

    tip_names = [
        "thumb-tip",
        "index-finger-tip",
        "middle-finger-tip",
        "ring-finger-tip",
        "pinky-finger-tip",
    ]

    mcp_names = [
        "thumb-metacarpal",
        "index-finger-metacarpal",
        "middle-finger-metacarpal",
        "ring-finger-metacarpal",
        "pinky-finger-metacarpal",
    ]

    for name in tip_names:
        p = torch.tensor(kps[name]["p_isaac_xyz"], dtype=torch.float32)
        v = p - wrist_p
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-6)
        feats.append(v)

    for name in mcp_names:
        p = torch.tensor(kps[name]["p_isaac_xyz"], dtype=torch.float32)
        v = p - wrist_p
        v = v / torch.clamp(torch.linalg.norm(v), min=1e-6)
        feats.append(v)

    return torch.cat(feats, dim=0)


def build_target(sample):
    return torch.tensor(
        sample["robot_right_wrist_joint_values_roll_pitch_yaw"],
        dtype=torch.float32,
    )


class WristMapBigMLP(nn.Module):
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


def load_all_samples():
    all_samples = []

    for path in DATA_PATHS:
        data = json.loads(path.read_text())
        samples = data["samples"]
        print("Loaded", len(samples), "from", path)
        all_samples.extend(samples)

    clean = []
    for s in all_samples:
        if s.get("webxr_right_wrist_quat_isaac_wxyz") is None:
            continue
        if s.get("webxr_right_palm_quat_isaac_wxyz") is None:
            continue
        if s.get("webxr_right_palm_normal_isaac_xyz") is None:
            continue
        if len(s.get("webxr_right_keypoints", {})) < 20:
            continue
        if s.get("packet_age_sec") is not None and s["packet_age_sec"] > 0.30:
            continue
        clean.append(s)

    print("Total samples:", len(all_samples))
    print("Clean samples:", len(clean))
    return clean


def main():
    samples = load_all_samples()

    X = torch.stack([build_feature(s) for s in samples])
    Y = torch.stack([build_target(s) for s in samples])

    print("X shape:", tuple(X.shape))
    print("Y shape:", tuple(Y.shape))

    x_mean = X.mean(dim=0)
    x_std = X.std(dim=0)
    x_std = torch.where(x_std < 1e-6, torch.ones_like(x_std), x_std)
    Xn = (X - x_mean) / x_std

    n = Xn.shape[0]
    perm = torch.randperm(n)

    n_train = int(n * 0.80)
    n_val = int(n * 0.10)

    train_idx = perm[:n_train]
    val_idx = perm[n_train:n_train + n_val]
    test_idx = perm[n_train + n_val:]

    X_train, Y_train = Xn[train_idx], Y[train_idx]
    X_val, Y_val = Xn[val_idx], Y[val_idx]
    X_test, Y_test = Xn[test_idx], Y[test_idx]

    model = WristMapBigMLP(in_dim=Xn.shape[1], out_dim=3)

    opt = optim.AdamW(model.parameters(), lr=5e-4, weight_decay=2e-3)
    loss_fn = nn.SmoothL1Loss(beta=0.08)

    best_val = float("inf")
    best_state = None
    patience = 700
    bad_epochs = 0

    for epoch in range(6000):
        model.train()

        pred = model(X_train)
        loss = loss_fn(pred, Y_train)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = loss_fn(val_pred, Y_val).item()

        if val_loss < best_val:
            best_val = val_loss
            bad_epochs = 0
            best_state = {
                "model": model.state_dict(),
                "x_mean": x_mean,
                "x_std": x_std,
                "input_dim": Xn.shape[1],
                "model_class": "WristMapBigMLP",
                "target_names": [
                    "right_wrist_roll_joint",
                    "right_wrist_pitch_joint",
                    "right_wrist_yaw_joint",
                ],
                "feature_description": [
                    "webxr_right_wrist_quat_isaac_wxyz",
                    "webxr_right_palm_quat_isaac_wxyz",
                    "webxr_right_palm_normal_isaac_xyz",
                    "unit wrist_to_thumb_tip",
                    "unit wrist_to_index_tip",
                    "unit wrist_to_middle_tip",
                    "unit wrist_to_ring_tip",
                    "unit wrist_to_pinky_tip",
                    "unit wrist_to_thumb_metacarpal",
                    "unit wrist_to_index_metacarpal",
                    "unit wrist_to_middle_metacarpal",
                    "unit wrist_to_ring_metacarpal",
                    "unit wrist_to_pinky_metacarpal",
                ],
            }
        else:
            bad_epochs += 1

        if epoch % 500 == 0 or epoch == 5999:
            with torch.no_grad():
                train_pred = model(X_train)
                val_pred = model(X_val)
                test_pred = model(X_test)

                train_mae = torch.abs(train_pred - Y_train).mean(dim=0)
                val_mae = torch.abs(val_pred - Y_val).mean(dim=0)
                test_mae = torch.abs(test_pred - Y_test).mean(dim=0)

                print(f"\nEpoch {epoch}")
                print("train loss:", round(loss.item(), 6), "val loss:", round(val_loss, 6), "best val:", round(best_val, 6))
                print("train MAE rad:", [round(float(x), 4) for x in train_mae])
                print("val   MAE rad:", [round(float(x), 4) for x in val_mae])
                print("test  MAE rad:", [round(float(x), 4) for x in test_mae])
                print("test  MAE deg:", [round(float(x * 180 / math.pi), 2) for x in test_mae])

        if bad_epochs >= patience:
            print("\nEarly stopping at epoch", epoch)
            break

    torch.save(best_state, OUT_PATH)

    print("\n====================================")
    print("Saved model:", OUT_PATH)
    print("Best val loss:", best_val)
    print("====================================")

    model.load_state_dict(best_state["model"])
    model.eval()

    with torch.no_grad():
        pred = model(X_test)
        err = torch.abs(pred - Y_test)

        mae = err.mean(dim=0)
        maxe = err.max(dim=0).values

        print("\nFINAL TEST")
        print("test MAE rad:", [round(float(x), 4) for x in mae])
        print("test MAE deg:", [round(float(x * 180 / math.pi), 2) for x in mae])
        print("test MAX rad:", [round(float(x), 4) for x in maxe])
        print("test MAX deg:", [round(float(x * 180 / math.pi), 2) for x in maxe])

        print("\nSample predictions:")
        for i in range(min(12, X_test.shape[0])):
            p = pred[i]
            y = Y_test[i]
            e = torch.abs(p - y)
            print(
                f"{i}: pred={[round(float(v), 3) for v in p]} "
                f"target={[round(float(v), 3) for v in y]} "
                f"err_deg={[round(float(v * 180 / math.pi), 1) for v in e]}"
            )


if __name__ == "__main__":
    main()
