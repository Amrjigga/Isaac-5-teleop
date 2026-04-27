# Isaac 5 Teleop — WebXR to G1 Inspire Hand/Wrist Retargeting

This repo contains the current best attempt at WebXR hand teleoperation / retargeting for the Unitree G1 Inspire hand in Isaac Lab / Isaac Sim.

The main goal is:

```text
WebXR hand tracking → robot wrist/palm orientation + fingers

The best current result is the 400-sample learned right-wrist model + split architecture, which visually improved right palm orientation, especially:

palm facing sky
palm facing floor

The split architecture also reduced wrist-position drift compared with the earlier learned-wrist version.

Current Best Result

Best live test script:

scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py

Best trained model:

models/right_wrist_mapping_model_400_big.pt

Best datasets:

data/g1_webxr_right_random_calibration_100.json
data/g1_webxr_right_random_calibration_300.json

Run the current best test inside ~/IsaacLab_5:

cd ~/IsaacLab_5
./isaaclab.sh -p scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py --device cpu
Environment / Setup Used

This was developed in:

Ubuntu / Linux laptop
Isaac Sim / Isaac Lab 5 setup
IsaacLab folder: ~/IsaacLab_5
Robot: Unitree G1 Inspire hand
Run style: ./isaaclab.sh -p <script.py> --device cpu

The scripts use the G1 Inspire config:

from isaaclab_assets.robots.unitree import G1_INSPIRE_FTP_CFG

The WebXR bridge sends hand tracking packets over UDP.

Expected hand joint format:

{
  "p": [x, y, z],
  "q": [qx, qy, qz, qw],
  "r": radius
}

Important coordinate conversion used:

# WebXR → Isaac approx
isaac_pos = [-z_web, -x_web, y_web]
WebXR Data Used

The model uses orientation and hand geometry, not absolute wrist position.

Logged data includes wrist position for debugging, but the model input is based on:

WebXR wrist quaternion
WebXR synthetic palm quaternion
WebXR palm normal
unit wrist-to-fingertip vectors
unit wrist-to-metacarpal vectors

Model target:

right_wrist_roll_joint
right_wrist_pitch_joint
right_wrist_yaw_joint

So the learned model predicts the robot wrist joint angles needed to imitate the observed WebXR hand/palm orientation.

Files Included
1. Calibration collection script
scripts/calibrate_webxr_to_g1_right_random_300.py

This script shows random robot right-wrist orientations. The user manually rotates their real right hand until the WebXR hand visually matches the robot palm, then presses Enter.

Run inside ~/IsaacLab_5:

cd ~/IsaacLab_5
./isaaclab.sh -p scripts/calibrate_webxr_to_g1_right_random_300.py --device cpu

Controls:

ENTER      save current matched pose and advance
s + ENTER  skip current pose
w + ENTER  write/save JSON now, keep going
q + ENTER  write/save JSON and quit

It auto-saves after each sample and auto-quits at 300 samples.

Output:

/tmp/g1_webxr_right_random_calibration_300.json
2. Training script
scripts/train_right_wrist_mapping_400_big.py

This trains the nonlinear right-wrist mapping model using:

data/g1_webxr_right_random_calibration_100.json
data/g1_webxr_right_random_calibration_300.json

Run inside ~/IsaacLab_5:

cd ~/IsaacLab_5
./isaaclab.sh -p scripts/train_right_wrist_mapping_400_big.py

Model architecture:

41 input features
→ 128
→ 128
→ 64
→ 3 wrist joint outputs

This is a nonlinear MLP using LayerNorm + GELU.

Output model:

right_wrist_mapping_model_400_big.pt

In this repo it is stored as:

models/right_wrist_mapping_model_400_big.pt
3. Split architecture live test
scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py

This is the current best live test.

Architecture:

right shoulder/elbow IK → wrist position
learned model → right wrist roll/pitch/yaw
fingers → current tracking

Why this matters:

Earlier learned-wrist tests improved palm orientation but caused too much wrist-position drift and elbow movement. The split version separates responsibilities:

arm = position
wrist = palm orientation
fingers = hand shape

Observed result:

~80% visually accurate overall
palm floor much better
palm sky much better
wrist position much better than previous learned version
some orientations still fail if not covered in training data
Calibration Datasets
100-sample dataset
data/g1_webxr_right_random_calibration_100.json

First learned-mapping dataset.

Quality check result:

100 samples
fresh packets
no missing wrist quaternion
no missing palm quaternion
all samples had enough keypoints
300-sample dataset
data/g1_webxr_right_random_calibration_300.json

Second wider-coverage dataset.

Coverage included:

mostly_pitch
mostly_roll
mostly_yaw
roll_pitch_diagonal
roll_pitch_opposite_diagonal
combined_random_wide
floor_sky_biased
neutral_reference

Quality check result:

300 samples
final=True
no skipped samples
unique packet counts=300
missing wrist quat=0
missing palm quat=0
samples with <20 keypoints=0
Training Result

The 400-sample model trained on:

100-sample dataset
+
300-sample dataset
=
400 total samples

Final test result:

test MAE rad: [0.2532, 0.256, 0.2542]
test MAE deg: [14.51, 14.67, 14.56]

test MAX rad: [0.8714, 0.7027, 0.9645]
test MAX deg: [49.92, 40.26, 55.26]

Interpretation:

Average error is usable visually.
Max error is still high on unseen/underrepresented orientations.
More targeted data is needed.
Known Issues
1. Some orientations still fail

The model is weak on orientations that were not represented enough in the data.

Fix:

collect more calibration samples around failed poses
especially diagonal/off-axis orientations
especially palm floor variants
2. WebXR frame vs robot frame

WebXR is headset/head-frame based, while the robot control frame is closer to torso/root.

Possible improvement:

calibrate WebXR/headset frame to robot torso frame
train/infer using torso-relative hand orientation
3. Left hand not trained yet

The current learned wrist model is right hand only.

Next step:

collect left-hand calibration data
train left wrist model
apply split architecture to left wrist too
Files in This Repo
scripts/
  calibrate_webxr_to_g1_right_random_300.py
  train_right_wrist_mapping_400_big.py
  quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py

data/
  g1_webxr_right_random_calibration_100.json
  g1_webxr_right_random_calibration_300.json

models/
  right_wrist_mapping_model_400_big.pt

docs/
  file_manifest.md
Recommended Next Steps
Collect more right-hand calibration samples around failed orientations.
Train a better model with more targeted data.
Train a left-hand model.
Improve WebXR/headset-frame to robot-torso-frame alignment.
Keep using split architecture because it improves wrist-position stability.
MD

## 2. Create file manifest

```bash
cd ~/Isaac-5-teleop

mkdir -p docs

cat > docs/file_manifest.md <<'MD'
# File Manifest

## scripts/calibrate_webxr_to_g1_right_random_300.py

Randomized right-hand orientation calibration script.

It shows robot right-wrist poses. The user manually matches their WebXR hand orientation to the robot palm orientation, presses Enter, and the script saves a matched sample.

Output:

```bash
/tmp/g1_webxr_right_random_calibration_300.json
scripts/train_right_wrist_mapping_400_big.py

Training script for the 400-sample nonlinear right-wrist mapping model.

Inputs:

data/g1_webxr_right_random_calibration_100.json
data/g1_webxr_right_random_calibration_300.json

Output:

right_wrist_mapping_model_400_big.pt
scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py

Live Isaac Lab split-architecture test.

Architecture:

right shoulder/elbow IK → wrist position
learned model → right wrist roll/pitch/yaw
fingers → current tracking
data/g1_webxr_right_random_calibration_100.json

First 100-sample right-hand calibration dataset.

data/g1_webxr_right_random_calibration_300.json

Second 300-sample right-hand calibration dataset with wider orientation coverage.

models/right_wrist_mapping_model_400_big.pt

Current best trained right-wrist model.
MD


## 3. Check files

```bash
cd ~/Isaac-5-teleop

find . -maxdepth 3 -type f | sort

You should see:

./README.md
./data/g1_webxr_right_random_calibration_100.json
./data/g1_webxr_right_random_calibration_300.json
./docs/file_manifest.md
./models/right_wrist_mapping_model_400_big.pt
./scripts/calibrate_webxr_to_g1_right_random_300.py
./scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py
./scripts/train_right_wrist_mapping_400_big.py


