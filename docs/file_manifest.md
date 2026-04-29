# File Manifest

## scripts/calibrate_webxr_to_g1_right_random_300.py
Randomized right-hand orientation calibration script. Shows robot wrist poses, waits for user to match WebXR hand orientation, then saves samples.

## scripts/train_right_wrist_mapping_400_big.py
Trains the nonlinear 400-sample right-wrist mapping model.

## scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_400_split.py
Live split-architecture teleop test:
- shoulder/elbow IK controls wrist position
- learned model controls right wrist roll/pitch/yaw
- fingers use current tracking

## data/g1_webxr_right_random_calibration_100.json
First 100-sample right-hand calibration dataset.

## data/g1_webxr_right_random_calibration_300.json
Second 300-sample right-hand calibration dataset with wider orientation coverage.

## models/right_wrist_mapping_model_400_big.pt
Current best trained right-wrist model.

---

## scripts/calibrate_webxr_to_g1_right_targeted_weakness_50.py

Targeted right-hand calibration script for the weak orientation region.

This focuses on poses where the 400-sample model was weak:

- palm facing floor on an angle
- palm angled forward
- palm angled up/forward
- negative roll + negative yaw combinations
- diagonal negative-roll poses

Output:

`/tmp/g1_webxr_right_targeted_weakness_calibration_50.json`

## data/g1_webxr_right_targeted_weakness_calibration_50.json

50-sample targeted weakness dataset.

Quality check:

- 50 samples
- final=True
- skipped=0
- unique packet counts=50
- missing wrist quat=0
- missing palm quat=0
- all samples had enough keypoints

Mode coverage:

- local_jitter_weakness: 23
- exact_weakness_pose: 12
- negative_roll_wide_family: 10
- bridge_to_normal: 5

## scripts/train_right_wrist_mapping_450_targeted.py

Training script using all right-hand calibration data:

- 100 general samples
- 300 wide general samples
- 50 targeted weakness samples
- 450 total samples

Output model:

`right_wrist_mapping_model_450_targeted.pt`

## models/right_wrist_mapping_model_450_targeted.pt

450-sample targeted model.

Numerically, this was slightly worse overall than the 400 model:

- 450-targeted test MAE deg: [15.21, 16.47, 16.71]
- 400-big test MAE deg: [14.51, 14.67, 14.56]

But visually, it was much better on the weak palm-orientation region:

- angled palm floor
- angled palm forward/up
- negative-roll diagonal poses

It is slightly worse on some normal/general poses, but good enough to keep.

## scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_450_split.py

Live split-architecture visual test using the 450-targeted model.

Run inside IsaacLab:

`./isaaclab.sh -p scripts/quest_g1_diffik_bimanual_viz_learned_right_wrist_450_split.py --device cpu`

Observed result:

- much better on previously weak orientations
- slightly worse on the rest
- still good enough to keep as a separate comparison script

Recommended use:

- `400_split` = better general model
- `450_split` = better weak-region model
---

## Left Palm Orientation Update

### scripts/calibrate_webxr_to_g1_left_orientation_450.py

Left-hand palm/wrist orientation calibration script.

It shows programmed left robot wrist/palm poses. The user visually matches their real left hand in WebXR, then presses Enter to save the sample.

The intended full dataset size is 450 poses, but the current saved dataset used for training is 300 samples.

Pose coverage includes:

- palm facing floor/down with angles
- palm facing sky/up with angles
- palm facing forward with angles
- palm facing sides with angles
- diagonal and random wrist orientations

Controls:

- ENTER = save current matched pose and advance
- s + ENTER = skip current pose
- w + ENTER = write/save JSON now and keep going
- q + ENTER = write/save JSON and quit

Output during collection:

- /tmp/g1_webxr_left_calibration_450.json

Current copied training dataset:

- data/g1_webxr_left_calibration_300.json

### data/g1_webxr_left_calibration_300.json

Left-hand calibration dataset.

Contains 300 matched samples.

Each sample includes:

- robot_left_wrist_joint_values_roll_pitch_yaw
- robot_left_wrist_yaw_link_pos_w
- robot_left_wrist_yaw_link_quat_wxyz
- webxr_left_wrist_quat_isaac_wxyz
- webxr_left_palm_quat_isaac_wxyz
- webxr_left_palm_normal_isaac_xyz
- webxr_left_keypoints

This dataset trains the left palm/wrist orientation model.

### scripts/train_left_wrist_mapping_300.py

Training script for the left wrist orientation model.

Input:

- data/g1_webxr_left_calibration_300.json

Output:

- left_wrist_mapping_model_300.pt

The model uses the same feature style as the right hand:

- WebXR wrist quaternion
- WebXR palm quaternion
- WebXR palm normal
- wrist-to-fingertip unit vectors
- wrist-to-metacarpal unit vectors

Target:

- left_wrist_roll_joint
- left_wrist_pitch_joint
- left_wrist_yaw_joint

Model type:

- nonlinear MLP
- 41 input features -> 128 -> 128 -> 64 -> 3 wrist joint outputs

Training result:

- test MAE deg: [12.39, 10.65, 10.25]
- test MAX deg: [56.62, 41.99, 43.28]

### models/left_wrist_mapping_model_300.pt

Trained left palm/wrist orientation model.

This is the current left-hand model trained on 300 left-hand calibration samples.

### scripts/quest_g1_diffik_bimanual_viz_left300_right450_split.py

Current bimanual learned wrist split-architecture script.

It loads:

- left_wrist_mapping_model_300.pt for the left wrist
- right_wrist_mapping_model_450_targeted.pt for the right wrist

Architecture:

- left/right shoulder-elbow IK controls wrist position
- learned wrist models control wrist roll/pitch/yaw
- finger tracking remains separate

Run inside IsaacLab:

./isaaclab.sh -p scripts/quest_g1_diffik_bimanual_viz_left300_right450_split.py --device cpu

Current status:

- left model works
- right 450 targeted model works
- script loads both hand models
- bimanual palm orientation pipeline is now active
