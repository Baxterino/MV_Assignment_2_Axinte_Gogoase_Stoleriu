import os
import sys
import time
import json
import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from mvlib import MarkerDetector, marker_pose, relative_pose, expm_so3, rot_error_deg

CALIB_FILE = "camera_calib.npz"
L_M = 0.092          # 92 mm masurati de tine
ID_F = 0
ID_M = 1
DICT_ID = cv2.aruco.DICT_6X6_250
IMG_W, IMG_H = 160, 120
DATA_DIR = "t5_data"

calib = np.load(CALIB_FILE)
K_FULL = calib["K"].astype(np.float64)
DIST_FULL = calib["dist"].astype(np.float64)

NATIVE_W, NATIVE_H = 640, 480
SX, SY = IMG_W / NATIVE_W, IMG_H / NATIVE_H
K_NET = np.array([
    [K_FULL[0, 0] * SX, 0.0, K_FULL[0, 2] * SX],
    [0.0, K_FULL[1, 1] * SY, K_FULL[1, 2] * SY],
    [0.0, 0.0, 1.0]
], dtype=np.float64)


def generate_synthetic(n_samples=12000, z_range=(0.25, 1.20), seed=42):
    os.makedirs(DATA_DIR, exist_ok=True)
    rng = np.random.default_rng(seed)
    d = cv2.aruco.getPredefinedDictionary(DICT_ID)
    side = 240
    quiet = 0.25 * L_M
    pad = int(round(quiet / L_M * side))
    patch = cv2.copyMakeBorder(cv2.aruco.generateImageMarker(d, ID_M, side),
                               pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=255)
    ps = patch.shape[0]
    half = L_M / 2.0 + quiet
    phys = np.array([[-half, half, 0], [half, half, 0], [half, -half, 0], [-half, -half, 0]], dtype=np.float64)
    src = np.float32([[-0.5, -0.5], [ps - 0.5, -0.5], [ps - 0.5, ps - 0.5], [-0.5, ps - 0.5]])

    X = np.zeros((n_samples, IMG_H, IMG_W), dtype=np.uint8)
    Rs = np.zeros((n_samples, 3, 3), dtype=np.float32)
    ts = np.zeros((n_samples, 3), dtype=np.float32)

    print(f"Generez {n_samples} imagini sintetice (L={L_M*1000:.0f} mm)...")
    t0 = time.time()
    for i in range(n_samples):
        Z = float(np.exp(rng.uniform(np.log(z_range[0]), np.log(z_range[1]))))
        phi = rng.uniform(0, 2 * np.pi)
        a = rng.uniform(0, 2 * np.pi)
        tilt = np.radians(rng.uniform(0, 50.0))
        R = expm_so3(np.array([np.cos(a), np.sin(a), 0.0]) * tilt) @ np.diag([1.0, -1.0, -1.0]) @ expm_so3([0, 0, phi])

        margin = 0.75 * K_NET[0, 0] * L_M / Z + 4
        u = rng.uniform(min(margin, IMG_W / 2 - 2), max(IMG_W - 1 - margin, IMG_W / 2))
        v = rng.uniform(min(margin, IMG_H / 2 - 2), max(IMG_H - 1 - margin, IMG_H / 2))
        t = Z * np.linalg.solve(K_NET, [u, v, 1.0])

        ss = 2
        low = rng.normal(0, 1, (int(rng.integers(3, 10)), int(rng.integers(4, 14)))).astype(np.float32)
        bg = cv2.resize(low, (IMG_W * ss, IMG_H * ss), interpolation=cv2.INTER_CUBIC)
        img = rng.uniform(60, 210) + rng.uniform(5, 45) * bg
        for _ in range(int(rng.integers(0, 6))):
            x0, y0 = rng.integers(0, IMG_W * ss, 2)
            w, h = rng.integers(10, 100, 2)
            cv2.rectangle(img, (int(x0), int(y0)), (int(x0 + w), int(y0 + h)), float(rng.uniform(0, 255)), -1)

        P = (R @ phys.T).T + t
        uv = (K_NET @ P.T).T
        uv = uv[:, :2] / uv[:, 2:3]
        dst = np.float32(uv * ss + (ss - 1) / 2.0)
        Hm = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(patch.astype(np.float32), Hm, (IMG_W * ss, IMG_H * ss), flags=cv2.INTER_LINEAR)
        mask = cv2.warpPerspective(np.full_like(patch, 255), Hm, (IMG_W * ss, IMG_H * ss), flags=cv2.INTER_NEAREST)
        img = np.where(mask > 0, warped, img)

        img = img * rng.uniform(0.6, 1.3) + rng.uniform(-20, 20)
        img = cv2.GaussianBlur(img, (0, 0), rng.uniform(0.3, 1.1) * ss)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        img = img + rng.normal(0, rng.uniform(1, 7), img.shape)

        X[i] = np.clip(img, 0, 255).astype(np.uint8)
        Rs[i], ts[i] = R.astype(np.float32), t.astype(np.float32)

    np.savez_compressed(os.path.join(DATA_DIR, "synth_train.npz"), X=X[:-1000], R=Rs[:-1000], t=ts[:-1000])
    np.savez_compressed(os.path.join(DATA_DIR, "synth_val.npz"), X=X[-1000:], R=Rs[-1000:], t=ts[-1000:])
    print(f"Gata in {time.time() - t0:.1f}s!")


def record_real_set(filename, max_frames=500):
    os.makedirs(DATA_DIR, exist_ok=True)
    det = MarkerDetector(DICT_ID, subpixel=True)
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)

    X_list, R_list, t_list, full_frames, T_CF_list = [], [], [], [], []
    recording = False
    print(f"=== COLECTARE: {filename} ===")
    print("Apasa [SPACE] = Start/Pauza | [S] = Salveaza | [Q] = Renunta")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = det.detect(gray)
        disp = frame.copy()

        if ID_M in found:
            cv2.aruco.drawDetectedMarkers(disp, [found[ID_M][None, :, :]], np.array([[ID_M]]))
        if ID_F in found:
            cv2.aruco.drawDetectedMarkers(disp, [found[ID_F][None, :, :]], np.array([[ID_F]]))

        if recording and ID_M in found:
            T_CM, rms, amb = marker_pose(found[ID_M], K_FULL, DIST_FULL, L_M)
            if rms < 1.5 and amb > 1.5:
                small_gray = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                X_list.append(small_gray)
                R_list.append(T_CM[:3, :3].astype(np.float32))
                t_list.append(T_CM[:3, 3].astype(np.float32))
                full_frames.append(gray)
                if ID_F in found:
                    T_CF, _, _ = marker_pose(found[ID_F], K_FULL, DIST_FULL, L_M)
                    T_CF_list.append(T_CF.astype(np.float32))
                else:
                    T_CF_list.append(np.full((4, 4), np.nan, dtype=np.float32))

        status = f"REC: {len(X_list)}/{max_frames}" if recording else f"PAUZA ({len(X_list)} cadre)"
        cv2.putText(disp, status, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0) if recording else (0, 200, 255), 2)
        cv2.imshow("Colectare Date", disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord(" "):
            recording = not recording
        elif key == ord("s") or len(X_list) >= max_frames:
            break
        elif key == ord("q"):
            cap.release()
            cv2.destroyAllWindows()
            return

    cap.release()
    cv2.destroyAllWindows()

    if len(X_list) > 0:
        out_path = os.path.join(DATA_DIR, filename)
        np.savez_compressed(
            out_path,
            X=np.array(X_list, dtype=np.uint8),
            R=np.array(R_list, dtype=np.float32),
            t=np.array(t_list, dtype=np.float32),
            full_gray=np.array(full_frames, dtype=np.uint8),
            T_CF=np.array(T_CF_list, dtype=np.float32)
        )
        print(f"Salvat {len(X_list)} cadre in {out_path}")


class PoseCNN(nn.Module):
    def __init__(self):
        super().__init__()
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, IMG_H), torch.linspace(-1, 1, IMG_W), indexing="ij")
        self.register_buffer("coords", torch.stack([xx, yy], dim=0).unsqueeze(0))

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=1), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 5, 256),
            nn.ReLU(),
            nn.Linear(256, 9)
        )
        with torch.no_grad():
            self.head[-1].bias.copy_(torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, np.log(0.6)], dtype=torch.float32))

    def forward(self, x_uint8):
        x = (x_uint8.float() / 255.0 - 0.5) / 0.25
        x = x.unsqueeze(1)
        coords = self.coords.expand(x.shape[0], -1, -1, -1)
        return self.head(self.features(torch.cat([x, coords], dim=1)))


def six_d_to_R_torch(a):
    a1, a2 = a[:, 0:3], a[:, 3:6]
    b1 = a1 / (torch.norm(a1, dim=1, keepdim=True) + 1e-8)
    a2 = a2 - torch.sum(b1 * a2, dim=1, keepdim=True) * b1
    b2 = a2 / (torch.norm(a2, dim=1, keepdim=True) + 1e-8)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack([b1, b2, b3], dim=2)


def decode_pose_torch(out):
    R = six_d_to_R_torch(out[:, :6])
    Z = torch.exp(out[:, 8])
    t = torch.stack([out[:, 6] * Z, out[:, 7] * Z, Z], dim=1)
    return R, t


def train_model(epochs_synth=15, epochs_real=12, batch_size=64, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    synth_d = np.load(os.path.join(DATA_DIR, "synth_train.npz"))
    X_s, R_s, t_s = synth_d["X"], synth_d["R"], synth_d["t"]
    Y_s = np.stack([t_s[:, 0] / t_s[:, 2], t_s[:, 1] / t_s[:, 2], np.log(t_s[:, 2])], axis=1).astype(np.float32)

    loader_synth = DataLoader(TensorDataset(torch.from_numpy(X_s), torch.from_numpy(R_s), torch.from_numpy(Y_s)),
                              batch_size=batch_size, shuffle=True)

    model = PoseCNN().to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs_synth)
    t0 = time.time()

    print("--- Etapa 1: Antrenare pe date sintetice ---")
    for ep in range(epochs_synth):
        model.train()
        tot_l = 0.0
        for xb, Rb, yb in loader_synth:
            xb, Rb, yb = xb.to(device), Rb.to(device), yb.to(device)
            if np.random.rand() < 0.5:
                xb = torch.clamp(xb.float() * np.random.uniform(0.8, 1.2) + np.random.uniform(-15, 15), 0, 255).to(torch.uint8)
            out = model(xb)
            Rh = six_d_to_R_torch(out[:, :6])
            loss = torch.mean(torch.sum((Rh - Rb) ** 2, dim=(1, 2))) + torch.mean(torch.sum(torch.abs(out[:, 6:9] - yb), dim=1))
            opt.zero_grad()
            loss.backward()
            opt.step()
            tot_l += loss.item()
        sched.step()
        print(f"Epoca {ep+1:2d}/{epochs_synth} | Loss: {tot_l/len(loader_synth):.4f} | {(time.time()-t0)/60:.1f} min")

    real_path = os.path.join(DATA_DIR, "real_train.npz")
    if os.path.exists(real_path):
        print("\n--- Etapa 2: Fine-tuning pe date reale ---")
        real_d = np.load(real_path)
        X_r, R_r, t_r = real_d["X"], real_d["R"], real_d["t"]
        Y_r = np.stack([t_r[:, 0] / t_r[:, 2], t_r[:, 1] / t_r[:, 2], np.log(t_r[:, 2])], axis=1).astype(np.float32)

        idx_s = np.random.choice(len(X_s), min(len(X_s), len(X_r) * 2), replace=False)
        loader_mix = DataLoader(TensorDataset(
            torch.from_numpy(np.concatenate([X_r, X_s[idx_s]])),
            torch.from_numpy(np.concatenate([R_r, R_s[idx_s]])),
            torch.from_numpy(np.concatenate([Y_r, Y_s[idx_s]]))
        ), batch_size=32, shuffle=True)

        opt_ft = torch.optim.AdamW(model.parameters(), lr=lr * 0.2, weight_decay=1e-4)
        for ep in range(epochs_real):
            model.train()
            tot_l = 0.0
            for xb, Rb, yb in loader_mix:
                xb, Rb, yb = xb.to(device), Rb.to(device), yb.to(device)
                out = model(xb)
                Rh = six_d_to_R_torch(out[:, :6])
                loss = torch.mean(torch.sum((Rh - Rb) ** 2, dim=(1, 2))) + torch.mean(torch.sum(torch.abs(out[:, 6:9] - yb), dim=1))
                opt_ft.zero_grad()
                loss.backward()
                opt_ft.step()
                tot_l += loss.item()
            print(f"Fine-tune {ep+1:2d}/{epochs_real} | Loss: {tot_l/len(loader_mix):.4f}")

    train_min = (time.time() - t0) / 60.0
    n_params = sum(p.numel() for p in model.parameters())
    torch.save({"state_dict": model.state_dict(), "train_minutes": train_min, "n_params": n_params}, "t5_model.pt")
    print(f"\nModel salvat: 't5_model.pt' ({n_params:,} parametri, {train_min:.2f} min)")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "synth":
        generate_synthetic(12000)
    elif cmd == "record_train":
        record_real_set("real_train.npz", 500)
    elif cmd == "train":
        train_model()