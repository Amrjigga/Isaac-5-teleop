# Isaac 5 Teleop — WebXR to Unitree G1 Inspire Wrist/Palm Retargeting

This repo contains WebXR to Isaac Lab teleoperation experiments for the Unitree G1 Inspire hand.

Current focus:
- WebXR hand tracking
- wrist position tracking
- learned right palm/wrist orientation mapping
- G1 Inspire wrist joint control
- split architecture for better wrist-position stability

## Current Best Scripts

General 400-sample split model:
scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py

450 targeted weakness split model:
scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_450_split.py

The 400 split model is better generally.
The 450 targeted model is better on the weak orientation region:
- palm facing floor on an angle
- palm angled forward/up
- negative-roll diagonal orientations

Run the 450 split model inside IsaacLab:

cd ~/IsaacLab_5
./isaaclab.sh -p scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_450_split.py --device cpu

## Environment

Developed with:
- Ubuntu/Linux laptop
- Isaac Sim / Isaac Lab 5 setup
- IsaacLab folder: ~/IsaacLab_5
- Robot config: Unitree G1 Inspire
- Run style: ./isaaclab.sh -p <script.py> --device cpu

Robot config used:
from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

## WebXR Input

WebXR hand packets provide joints with:
- p: [x, y, z]
- q: [qx, qy, qz, qw]
- r: radius

Coordinate conversion used:
isaac_pos = [-z_web, -x_web, y_web]

The learned model does not train on absolute wrist position. It uses orientation and hand geometry:
- WebXR wrist quaternion
- WebXR synthetic palm quaternion
- WebXR palm normal
- wrist-to-fingertip unit vectors
- wrist-to-metacarpal unit vectors

Target output:
- right_wrist_roll_joint
- right_wrist_pitch_joint
- right_wrist_yaw_joint

## Data

Datasets:
- data/g1_webxr_right_random_calibration_100.json
- data/g1_webxr_right_random_calibration_300.json
- data/g1_webxr_right_targeted_weakness_calibration_50.json

Dataset sizes:
- 100-sample general right-hand calibration
- 300-sample wider right-hand calibration
- 50-sample targeted weak-pose calibration

Latest total:
100 + 300 + 50 = 450 samples

## Models

Models:
- models/right_wrist_mapping_model_400_big.pt
- models/right_wrist_mapping_model_450_targeted.pt

400 model:
test MAE deg: [14.51, 14.67, 14.56]

450 targeted model:
test MAE deg: [15.21, 16.47, 16.71]

The 450 model is numerically slightly worse overall, but visually better on the targeted weak orientations.

## Training Scripts

Training scripts:
- scripts/train_right_wrist_mapping_400_big.py
- scripts/train_right_wrist_mapping_450_targeted.py

Model type:
- nonlinear MLP
- 41 input features -> 128 -> 128 -> 64 -> 3 wrist joint outputs

## Calibration Scripts

Calibration scripts:
- scripts/calibrate_webxr_to_g1_right_random_300.py
- scripts/calibrate_webxr_to_g1_right_targeted_weakness_50.py

Calibration flow:
1. Robot shows a programmed/random right wrist orientation.
2. User rotates real right hand in WebXR to visually match the robot palm.
3. User presses Enter to save the matched sample.
4. JSON dataset is written.

## Split Architecture

The split architecture separates position and orientation:

right shoulder/elbow IK -> wrist position
learned model -> right wrist roll/pitch/yaw
fingers -> current tracking

This reduced wrist position drift compared with the earlier learned-wrist script.

## Recommended Next Steps

1. Compare 400 split vs 450 split visually.
2. Collect more targeted samples around failed orientations.
3. Train left-hand model.
4. Improve WebXR/headset-frame to robot-torso-frame alignment.
5. Continue using split architecture for wrist-position stability.
