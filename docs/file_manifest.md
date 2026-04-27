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
