"""
make_demo_video.py
Generates a ~2-minute demonstration video (demo_video.mp4) combining:
  - Part 1: Live camera tracking with ArUco corners & 3D axes (from t4_camera_video.mp4)
  - Part 2: Synchronized side-by-side T4 camera video + 3D trajectory GIF (t4_relative_pose.gif)
  - Part 3: Live side-by-side comparison of Classical vs. PoseCNN
            (highlighting robustness under corner occlusion and motion blur)

Usage:
  python make_demo_video.py --mode live
  python make_demo_video.py --mode offline
"""

import argparse
import os
import time
import cv2
import numpy as np
import torch
from PIL import Image, ImageSequence

from t5_pipeline import PoseCNN

L_M = 0.092  # 92 mm physical marker size
OUT_W, OUT_H = 1280, 540
FPS = 30.0


def rot6d_to_matrix_np(d6):
    """Converts a 6D rotation vector (6,) into a 3x3 rotation matrix via Gram-Schmidt."""
    a1 = d6[:3]
    a2 = d6[3:6]
    b1 = a1 / (np.linalg.norm(a1) + 1e-8)
    b2 = a2 - np.dot(b1, a2) * b1
    b2 = b2 / (np.linalg.norm(b2) + 1e-8)
    b3 = np.cross(b1, b2)
    return np.stack([b1, b2, b3], axis=-1)


def load_calib_npz(path="camera_calib.npz"):
    data = np.load(path)
    K = data["K"] if "K" in data else data["camera_matrix"]
    dist = data["dist"] if "dist" in data else data["dist_coeffs"]
    return K.astype(np.float64), dist.astype(np.float64)


def get_aruco_detector():
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    params = cv2.aruco.DetectorParameters()
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    return cv2.aruco.ArucoDetector(aruco_dict, params)


def estimate_pose_ippe(corners_2d, L, K, dist):
    half = L / 2.0
    obj_pts = np.array([
        [-half,  half, 0.0],
        [ half,  half, 0.0],
        [ half, -half, 0.0],
        [-half, -half, 0.0],
    ], dtype=np.float64)
    img_pts = corners_2d.reshape(4, 2).astype(np.float64)

    ok, rvecs, tvecs, _ = cv2.solvePnPGeneric(
        obj_pts, img_pts, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if not ok or len(rvecs) == 0:
        return None

    rvec, tvec = cv2.solvePnPRefineLM(obj_pts, img_pts, K, dist, rvecs[0], tvecs[0])
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.reshape(3)
    return T


def draw_axes_T(img, K, dist, T, length=0.05):
    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    cv2.drawFrameAxes(img, K, dist, rvec, tvec, length, 2)


def draw_axes_safe(img, K, dist, T, length=0.07):
    """Projects 3D origin + X/Y/Z axes onto the image with a dark border for high visibility."""
    pts_3d = np.array([
        [0.0, 0.0, 0.0],
        [length, 0.0, 0.0],  # X - Red
        [0.0, length, 0.0],  # Y - Green
        [0.0, 0.0, length],  # Z - Blue
    ], dtype=np.float64)

    rvec, _ = cv2.Rodrigues(T[:3, :3])
    tvec = T[:3, 3].reshape(3, 1)
    pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
    pts_2d = np.int32(np.clip(pts_2d.reshape(-1, 2), -2000, 2000))

    origin = tuple(pts_2d[0])
    for idx, color in zip([1, 2, 3], [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        end = tuple(pts_2d[idx])
        cv2.line(img, origin, end, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.line(img, origin, end, color, 3, cv2.LINE_AA)
    cv2.circle(img, origin, 6, (0, 0, 0), -1)
    cv2.circle(img, origin, 4, (0, 255, 255), -1)


def make_banner(frame, title, subtitle="", color=(30, 58, 138)):
    h, w = frame.shape[:2]
    bar_h = 54
    canvas = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    canvas[:bar_h, :] = color
    canvas[bar_h:, :] = frame
    cv2.putText(canvas, title, (18, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.68, (255, 255, 255), 2, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle, (18, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 230, 255), 1, cv2.LINE_AA)
    return canvas


def load_gif_frames(gif_path):
    pil_gif = Image.open(gif_path)
    frames = []
    for frame in ImageSequence.Iterator(pil_gif):
        rgb = np.array(frame.convert("RGB"))
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        frames.append(bgr)
    return frames


def load_mp4_frames(mp4_path):
    cap = cv2.VideoCapture(mp4_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames


def build_part1_and_part2(writer, cam_frames, gif_frames):
    panel_w, panel_h = OUT_W // 2, OUT_H - 54

    # Part 1: First 15 seconds showing ArUco detection & 3D axes
    p1_count = min(len(cam_frames), int(15 * FPS))
    for i in range(p1_count):
        f = cv2.resize(cam_frames[i], (OUT_W, OUT_H - 54))
        out = make_banner(
            f,
            "Part 1: Real-Time ArUco Detection & 6-DoF Pose Axes (T1-T3)",
            "DICT_6X6_250 | L = 92.0 mm | Sub-pixel corners + IPPE-Square + LM Refinement",
        )
        writer.write(out)

    # Part 2: Synchronized side-by-side T4 camera video + 3D Matplotlib animation (30s)
    target_p2_frames = int(30 * FPS)
    for k in range(target_p2_frames):
        cam_idx = min(int(k * len(cam_frames) / target_p2_frames), len(cam_frames) - 1)
        gif_idx = min(int(k * len(gif_frames) / target_p2_frames), len(gif_frames) - 1)
        left = cv2.resize(cam_frames[cam_idx], (panel_w, panel_h))
        right = cv2.resize(gif_frames[gif_idx], (panel_w, panel_h))
        combined = np.hstack([left, right])
        out = make_banner(
            combined,
            "Part 2: Synchronized Relative Pose Tracking F -> M (Task T4)",
            "Left: Camera Feed with Pose Overlays | Right: 3D Trajectory & Euler Angles",
        )
        writer.write(out)


def load_posecnn(model_path="t5_model.pt", device="cpu"):
    model = PoseCNN().to(device)
    ckpt = torch.load(model_path, map_location=device)
    state = ckpt["state_dict"] if isinstance(ckpt, dict) and "state_dict" in ckpt else ckpt
    model.load_state_dict(state)
    model.eval()
    return model


@torch.no_grad()
def predict_nn_pose(model, gray_frame, device="cpu"):
    # Use exact [0, 1] scaling from t5_pipeline.py
    img = cv2.resize(gray_frame, (160, 120)).astype(np.float32) / 255.0
    tensor_in = torch.from_numpy(img[None, ...]).to(device)  # (1, 120, 160) -> PoseCNN unsqueezes & adds coords

    out = model(tensor_in)[0].cpu().numpy()
    R = rot6d_to_matrix_np(out[:6])
    u_n, v_n, log_z = out[6], out[7], out[8]
    Z = float(np.clip(np.exp(log_z), 0.15, 2.0))
    t = np.array([u_n * Z, v_n * Z, Z], dtype=np.float64)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def record_part3_live(writer, K, dist, model, duration_s=80.0):
    detector = get_aruco_detector()
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    panel_w, panel_h = OUT_W // 2, OUT_H - 54
    print(f"\n[Part 3] Recording {duration_s:.0f}s side-by-side comparison (Classical vs. PoseCNN)...")
    print("Suggested sequence during recording:")
    print("  0-25s : Move marker M normally in 3D space (both methods track)")
    print("  25-55s: Cover 1 corner of marker M with your finger/card (Classical drops out, PoseCNN continues)")
    print("  55-80s: Move marker M rapidly / dim light (Motion blur robustness)")

    t0 = time.time()
    while time.time() - t0 < duration_s:
        ret, frame = cap.read()
        if not ret:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Left panel: Classical ArUco + IPPE
        left_vis = frame.copy()
        corners, ids, _ = detector.detectMarkers(gray)
        T_F, T_M_clas = None, None
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(left_vis, corners, ids)
            for c, mid in zip(corners, ids.flatten()):
                if mid == 0:
                    T_F = estimate_pose_ippe(c, L_M, K, dist)
                elif mid == 1:
                    T_M_clas = estimate_pose_ippe(c, L_M, K, dist)

        if T_F is not None:
            draw_axes_T(left_vis, K, dist, T_F, length=0.05)
        if T_M_clas is not None:
            draw_axes_T(left_vis, K, dist, T_M_clas, length=0.05)
            status_c = "CLASSICAL: DETECTED"
            col_c = (0, 220, 0)
            if T_F is not None:
                T_rel_c = np.linalg.inv(T_F) @ T_M_clas
                tx, ty, tz = T_rel_c[:3, 3] * 1000.0
                cv2.putText(left_vis, f"Rel t: [{tx:.0f}, {ty:.0f}, {tz:.0f}] mm", (15, 65),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, col_c, 2)
        else:
            status_c = "CLASSICAL: LOST (Occlusion / Blur)"
            col_c = (0, 0, 255)
        cv2.putText(left_vis, status_c, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, col_c, 2)

        # Right panel: PoseCNN direct regression
        right_vis = frame.copy()
        T_M_nn = predict_nn_pose(model, gray)
        # If PoseCNN outputs relative pose F->M (Z < 0.25m while F is visible), chain with T_F
        if T_F is not None and T_M_nn[2, 3] < 0.25:
            T_draw_nn = T_F @ T_M_nn
        else:
            T_draw_nn = T_M_nn
        draw_axes_safe(right_vis, K, dist, T_draw_nn, length=0.07)

        tx_n, ty_n, tz_n = T_M_nn[:3, 3] * 1000.0
        cv2.putText(right_vis, "PoseCNN: 100% CONTINUOUS OUTPUT", (15, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)
        cv2.putText(right_vis, f"Pred t: [{tx_n:.0f}, {ty_n:.0f}, {tz_n:.0f}] mm", (15, 65),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 200, 0), 2)

        elapsed = time.time() - t0
        combined = np.hstack([
            cv2.resize(left_vis, (panel_w, panel_h)),
            cv2.resize(right_vis, (panel_w, panel_h)),
        ])
        out = make_banner(
            combined,
            f"Part 3: Classical Geometric vs. PoseCNN Side-by-Side ({elapsed:.1f}s / {duration_s:.0f}s)",
            "Left: Sub-pixel ArUco + IPPE (High Precision, Fails on Corner Occlusion) | Right: PoseCNN (100% Availability)",
        )
        writer.write(out)
        cv2.imshow("Demo Recording (Part 3) - Press Q to finish early", out)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["live", "offline"], default="live")
    parser.add_argument("--out", default="demo_video.mp4")
    args = parser.parse_args()

    if not os.path.exists("t4_camera_video.mp4") or not os.path.exists("t4_relative_pose.gif"):
        raise FileNotFoundError("Ensure t4_camera_video.mp4 and t4_relative_pose.gif exist in D:\\mv_code\\code.")

    cam_frames = load_mp4_frames("t4_camera_video.mp4")
    gif_frames = load_gif_frames("t4_relative_pose.gif")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(args.out, fourcc, FPS, (OUT_W, OUT_H))

    print("Rendering Part 1 (Live Axes, 15s) and Part 2 (Synchronized T4 Side-by-Side, 30s)...")
    build_part1_and_part2(writer, cam_frames, gif_frames)

    if args.mode == "live":
        K, dist = load_calib_npz("camera_calib.npz")
        model = load_posecnn("t5_model.pt")
        record_part3_live(writer, K, dist, model, duration_s=80.0)

    writer.release()
    print(f"Done! Saved {args.out} (~2 min 05s) successfully.")


if __name__ == "__main__":
    main()