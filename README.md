# 3D Relative Pose Tracking Using ArUco Fiducial Markers: Classical Geometric PnP vs. Direct CNN Regression

**Authors:** Stefan Axinte, Elena Gogoase, George Stoleriu (MLRC II 1B)

A real-time monocular 6-DoF relative pose estimation system ($^F\mathbf{T}_M \in SE(3)$) tracking a mobile target marker ($M$, ID `1`) relative to a stationary world-reference marker ($F$, ID `0`). This repository implements, calibrates, and benchmarks two complementary paradigms across six standardized evaluation protocols (**E1–E6**):

1. **Classical Geometric Pipeline:** Sub-pixel ArUco corner detection (`DICT_6X6_250`) combined with Infinitesimal Plane-Based Pose Estimation (`SOLVEPNP_IPPE_SQUARE`) and Levenberg–Marquardt non-linear refinement.
2. **Deep Learning Pipeline (`PoseCNN`):** End-to-end Coordinate Convolutional Neural Network (`CoordConv`) operating on $160 \times 120$ grayscale frames with continuous 6D rotation parameterization (Gram–Schmidt orthogonalization) and perspective-consistent logarithmic depth ($\log Z$).

---

## Repository Structure

```text
├── README.md                 # Setup instructions and full reproduction guide
├── requirements.txt          # Python package dependencies
├── .gitignore                # Excludes large raw synthetic training archives and cache files
├── camera_calib.npz          # Calibrated camera intrinsics K and distortion d (RMS = 0.1298 px)
├── t5_model.pt               # Pre-trained PoseCNN weights (940,649 params, 3.83 MB)
├── mvlib.py                  # Core rigid-body transformations, ArUco detection, and metrics
├── calibrate_webcam.py       # T2: Interactive chessboard intrinsic camera calibration
├── track_relative_pose.py    # T3: Live 6-DoF relative pose estimation overlay
├── t4_live_and_record.py     # T4: Live 3D coordinate frame visualization and 20s recording
├── t5_pipeline.py            # T5: Synthetic dataset generation, real capture, and PoseCNN training
├── run_experiments.py        # T6: Standardized experimental benchmark suite (E1–E6)
└── make_demo_video.py        # Renders the 2-minute side-by-side demonstration video (demo_video.mp4)
```

---

## Hardware & Calibration Specifications

* **Camera:** Standard RGB Webcam operating at native $640 \times 480\text{ px}$ resolution.
* **Calibration Accuracy:** $\text{RMS} = 0.1298\text{ px}$ across $21$ chessboard viewpoints ($9 \times 6$ internal corners).
* **Intrinsic Matrix ($\mathbf{K}$):** $f_x = 556.0604\text{ px}$, $f_y = 556.0404\text{ px}$, $c_x = 319.8285\text{ px}$, $c_y = 261.1154\text{ px}$.
* **Distortion Coefficients ($\mathbf{d}$):** $[0.00173, \; 0.05059, \; 0.00124, \; 0.00141, \; -0.43341]^T$.
* **Fiducial Markers:** OpenCV `DICT_6X6_250` (Reference $F$: ID `0`, Target $M$: ID `1`), printed at 300 DPI and mounted on rigid planar cardboard with a verified black-border side length of **$L = 92.0\text{ mm}$ ($0.092\text{ m}$)**.

---

## Environment Setup

1. Clone this repository and navigate to the project directory:
   ```bash
   git clone <your-repository-url>
   cd aruco-relative-pose
   ```

2. Create a virtual environment and install the required dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux/macOS
   venv\Scripts\activate           # Windows PowerShell

   pip install -r requirements.txt
   ```

---

## Step-by-Step Reproduction Guide

All random number generators (`numpy`, `torch`, `random`) are initialized with fixed seeds (`seed=42`) to ensure deterministic dataset generation, reproducible training splits, and identical evaluation numbers.

### 1. Camera Calibration (Task T2)
*(Skip if using the included `camera_calib.npz`)*
```bash
python calibrate_webcam.py
```
Capture 20+ diverse viewpoints of a $9 \times 6$ chessboard (`SPACE` to capture, `C` to compute calibration).

### 2. Real-Time Tracking & Synchronized 3D Animation (Tasks T3–T4)
* **Live Pose Overlay (T3):**
  ```bash
  python track_relative_pose.py
  ```
* **3D Matplotlib Visualization & 20s Synchronized Recording (T4):**
  ```bash
  python t4_live_and_record.py
  ```
  Press `R` in the camera window to record a 20-second trajectory sequence. Outputs `t4_camera_video.mp4`, `t4_relative_pose.gif`, and `t4_recorded_seq.npz`.

### 3. Neural Network Data Generation & Training (Task T5)
*(Skip training if using the included `t5_model.pt` weights)*
```bash
# Step 3a: Generate 11,000 synthetic training + 1,000 validation images (seed=42)
python t5_pipeline.py synth

# Step 3b: Record 500 real-world fine-tuning frames using the webcam
python t5_pipeline.py record_train

# Step 3c: Two-stage training (15 epochs synthetic + 12 epochs real fine-tuning)
python t5_pipeline.py train
```

### 4. Experimental Benchmarks (Task T6: E1–E6)
Run each evaluation protocol directly against the saved calibration and model weights:

```bash
# E1: Static accuracy & temporal jitter (5 ruler positions x 100 frames)
python run_experiments.py e1

# E2: Distance & perspective tilt sweep (40 cm / 0 deg, 70 cm / 30 deg, 100 cm / 50 deg)
python run_experiments.py e2

# E3: Dynamic 20-second linear trajectory tracking along a 300 mm ruler
python run_experiments.py e3

# E4 & E6: Robustness stress tests (motion blur, dim light, corner occlusion) + FPS/memory cost
python run_experiments.py e4

# E5: Offline evaluation on the 1,000-frame synthetic validation set (seed=42)
python run_experiments.py e5
```

### 5. Demonstration Video Generation
To reproduce the side-by-side comparison video (`demo_video.mp4`) combining T1–T3 overlays, the synchronized T4 animation, and live Classical vs. `PoseCNN` tracking:
```bash
python make_demo_video.py --mode live
```

---

## Summary of Benchmark Results (E1–E6)

Values are reported as **Median (95th Percentile)**:

| Metric / Experiment | Classical (`ArUco + IPPE + LM`) | `PoseCNN` (`6D + log Z`) |
| :--- | :---: | :---: |
| **E1: Static Translation Error $e_t$ [mm]** | **15.9 (21.5)** | 183.3 (208.5) |
| **E1: Static Rotation Error $e_R$ [deg]** | **3.6 (14.1)** | 31.4 (53.2) |
| **E1: Static Jitter $\sigma_t$ [mm]** | **7.73** | 14.30 |
| **E2: Range & Tilt Detection (40/70/100 cm)** | 98% / 100% / 100% | **100% / 100% / 100%** |
| **E3: Dynamic Trajectory $e_t$ / RMSE [mm]** | 238.1 (255.5) / 240.4 | 136.6 (216.6) / 139.9 |
| **E4: Robustness Output Rate (Mean)** | 33% *(Blur: 33%, Dim: 64%, Occ: 3%)* | **100%** *(100% all scenarios)* |
| **E5: Synthetic $e_t$ [mm] / $e_R$ [deg]** | 378.7 (756.9) / **23.4 (54.7)** | **46.8 (219.1)** / 33.0 (73.9) |
| **E5: Synthetic Detection Rate** | 74.2% | **100.0%** |
| **E6: Inference Throughput [FPS]** | **885.4 FPS** (1.13 ms) | **486.6 FPS** (2.05 ms) |
| **E6: Storage & Training Overhead** | **0 KB / 0 min** | 3,831 KB / 4.71 min |