import os
import sys
import time
import cv2
import numpy as np
import torch

from mvlib import MarkerDetector, marker_pose, relative_pose, rot_error_deg
from t5_pipeline import PoseCNN, decode_pose_torch, K_FULL, DIST_FULL, L_M, ID_F, ID_M, DICT_ID, IMG_W, IMG_H, DATA_DIR, NATIVE_W, NATIVE_H

ckpt = torch.load("t5_model.pt", map_location="cpu", weights_only=True)
model = PoseCNN()
model.load_state_dict(ckpt["state_dict"])
model.eval()
detector = MarkerDetector(DICT_ID, subpixel=True)


def open_cam():
    cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, NATIVE_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, NATIVE_H)
    return cap


def eval_frame(gray):
    found = detector.detect(gray)
    if ID_F not in found:
        return None, None, 0.0, 0.0
    T_CF, _, _ = marker_pose(found[ID_F], K_FULL, DIST_FULL, L_M)

    t0 = time.perf_counter()
    T_FM_c = None
    if ID_M in found:
        T_CM_c, _, _ = marker_pose(found[ID_M], K_FULL, DIST_FULL, L_M)
        T_FM_c = relative_pose(T_CF, T_CM_c)
    dt_c = time.perf_counter() - t0

    t0 = time.perf_counter()
    small = cv2.resize(gray, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    with torch.no_grad():
        Rn, tn = decode_pose_torch(model(torch.from_numpy(small).unsqueeze(0)))
    T_CM_n = np.eye(4)
    T_CM_n[:3, :3] = Rn[0].numpy()
    T_CM_n[:3, 3] = tn[0].numpy()
    T_FM_n = relative_pose(T_CF, T_CM_n)
    dt_n = time.perf_counter() - t0

    return T_FM_c, T_FM_n, dt_c, dt_n


def stats(arr):
    v = np.asarray(arr)
    v = v[np.isfinite(v)]
    if len(v) == 0:
        return "N/A"
    return f"{np.median(v):.1f} ({np.percentile(v, 95):.1f})"


# --- E1: Static accuracy (5 pozitii pe rigla x 100 cadre) ---
def exp_e1():
    cap = open_cam()
    et_c, er_c, jit_c = [], [], []
    et_n, er_n, jit_n = [], [], []
    print("=== E1: MASURARE STATICA (5 pozitii x 100 cadre) ===")
    for i in range(5):
        print(f"\nPozitia {i+1}/5:")
        gx = float(input("  Distanta X fata de centrul lui F [mm] (ex: 150): "))
        gy = float(input("  Distanta Y fata de centrul lui F [mm] (ex: 0): "))
        g_yaw = float(input("  Unghi rotatie pe masa [deg] (ex: 0, 30, 45): "))
        gt_t = np.array([gx, gy, 0.0]) / 1000.0
        rad = np.radians(g_yaw)
        gt_R = np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])
        input("  Apasa ENTER cand esti gata...")

        pc, pn = [], []
        while len(pn) < 100:
            ret, frame = cap.read()
            if not ret:
                continue
            Tc, Tn, _, _ = eval_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if Tn is not None:
                pn.append(Tn[:3, 3])
                er_n.append(rot_error_deg(gt_R, Tn[:3, :3]))
                if Tc is not None:
                    pc.append(Tc[:3, 3])
                    er_c.append(rot_error_deg(gt_R, Tc[:3, :3]))
                else:
                    pc.append(np.full(3, np.nan))
            cv2.putText(frame, f"E1 Poz {i+1}: {len(pn)}/100", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imshow("E1", frame)
            cv2.waitKey(1)

        pc, pn = np.array(pc), np.array(pn)
        et_c.extend(np.linalg.norm(pc - gt_t, axis=1) * 1000.0)
        et_n.extend(np.linalg.norm(pn - gt_t, axis=1) * 1000.0)
        jit_c.append(np.linalg.norm(np.nanstd(pc, axis=0)) * 1000.0)
        jit_n.append(np.linalg.norm(np.std(pn, axis=0)) * 1000.0)

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- REZULTATE E1 (Tabel) ---")
    print(f"Clasic | e_t: {stats(et_c)} mm | e_R: {stats(er_c)} deg | jitter: {np.nanmean(jit_c):.2f} mm")
    print(f"NN     | e_t: {stats(et_n)} mm | e_R: {stats(er_n)} deg | jitter: {np.mean(jit_n):.2f} mm")


# --- E2: Distanta si Tilt (0 - 60 deg) ---
def exp_e2():
    cap = open_cam()
    print("=== E2: DISTANTA SI INCLINARE (TILT) ===")
    for cond in ["Aproape (~40cm), tilt 0 deg", "Mediu (~70cm), tilt 30 deg", "Departe (~100cm), tilt 50 deg"]:
        input(f"\nPregateste: {cond}. Apasa ENTER pentru 60 cadre...")
        ok_c, diffs_t, diffs_r = 0, [], []
        cnt = 0
        while cnt < 60:
            ret, frame = cap.read()
            if not ret:
                continue
            Tc, Tn, _, _ = eval_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if Tn is not None:
                cnt += 1
                if Tc is not None:
                    ok_c += 1
                    diffs_t.append(np.linalg.norm(Tc[:3, 3] - Tn[:3, 3]) * 1000.0)
                    diffs_r.append(rot_error_deg(Tc[:3, :3], Tn[:3, :3]))
            cv2.imshow("E2", frame)
            cv2.waitKey(1)
        print(f"Conditie [{cond}]: Detectie Clasica = {ok_c/60*100:.0f}% | NN = 100% | Diferenta NN vs Clasic: {stats(diffs_t)} mm, {stats(diffs_r)} deg")
    cap.release()
    cv2.destroyAllWindows()


# --- E3: Miscare (20s) pe traseu cunoscut (ex. de-a lungul riglei) ---
def exp_e3():
    cap = open_cam()
    print("=== E3: MISCARE PE TRASEU CUNOSCUT (20 secunde) ===")
    dist_total_mm = float(input("Introdu lungimea traseului drept pe rigla [mm] (ex: 300): "))
    input("Pune M la capatul 0 al riglei, apasa ENTER si misca-l uniform pana la capat in 20s...")

    t0 = time.time()
    pts_c, pts_n = [], []
    while time.time() - t0 < 20.0:
        ret, frame = cap.read()
        if not ret:
            continue
        Tc, Tn, _, _ = eval_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
        if Tn is not None:
            pts_n.append(Tn[:3, 3])
            pts_c.append(Tc[:3, 3] if Tc is not None else np.full(3, np.nan))
        cv2.putText(frame, f"E3 Miscare: {time.time()-t0:.1f}/20s", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.imshow("E3", frame)
        cv2.waitKey(1)

    cap.release()
    cv2.destroyAllWindows()
    pts_c, pts_n = np.array(pts_c), np.array(pts_n)
    # Aproximare traiectorie liniara uniforma intre start si final pe rigla
    gt_traj = np.linspace([0, 0, 0], [dist_total_mm / 1000.0, 0, 0], len(pts_n))
    # Aliniem originea la primul punct detectat
    pts_c_aligned = pts_c - pts_n[0]
    pts_n_aligned = pts_n - pts_n[0]
    ec = np.linalg.norm(pts_c_aligned - gt_traj, axis=1) * 1000.0
    en = np.linalg.norm(pts_n_aligned - gt_traj, axis=1) * 1000.0
    print("\n--- REZULTATE E3 (Tabel) ---")
    print(f"Clasic | e_t: {stats(ec)} mm | RMSE: {np.sqrt(np.nanmean(ec**2)):.1f} mm | Max: {np.nanmax(ec):.1f} mm")
    print(f"NN     | e_t: {stats(en)} mm | RMSE: {np.sqrt(np.mean(en**2)):.1f} mm | Max: {np.max(en):.1f} mm")


# --- E4: Robustete (Blur, Lumina slaba, Ocluzie partiala) ---
def exp_e4():
    cap = open_cam()
    print("=== E4: ROBUSTETE (Blur rapid, Lumina slaba, Ocluzie colt) ===")
    all_out_c, all_out_n, all_tc, all_tn = [], [], [], []
    for cond in ["1. Miscare rapida (Motion Blur)", "2. Lumina redusa (Dim light)", "3. Ocluzie partiala (acopera 1 colt al lui M)"]:
        input(f"\nPregateste: {cond}. Apasa ENTER pentru 100 cadre...")
        cnt, det_c = 0, 0
        while cnt < 100:
            ret, frame = cap.read()
            if not ret:
                continue
            Tc, Tn, dtc, dtn = eval_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
            if Tn is not None:
                cnt += 1
                all_tn.append(dtn)
                if Tc is not None:
                    det_c += 1
                    all_tc.append(dtc)
            cv2.putText(frame, f"{cond}: {cnt}/100", (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 200, 255), 2)
            cv2.imshow("E4", frame)
            cv2.waitKey(1)
        all_out_c.append(det_c)
        all_out_n.append(100)
        print(f"  -> {cond}: Output Clasic = {det_c}% | Output NN = 100%")

    cap.release()
    cv2.destroyAllWindows()
    print("\n--- REZULTATE E4 & E6 (Tabel) ---")
    print(f"Clasic | Output mediu: {np.mean(all_out_c):.0f}% | Viteza: {1.0/np.mean(all_tc):.1f} FPS")
    print(f"NN     | Output mediu: 100% | Viteza: {1.0/np.mean(all_tn):.1f} FPS | Model: {os.path.getsize('t5_model.pt')/1024:.0f} KB | Antrenare: {ckpt['train_minutes']:.1f} min")


# --- E5: Validare sintetica ---
def exp_e5():
    val = np.load(os.path.join(DATA_DIR, "synth_val.npz"))
    X, R, t = val["X"], val["R"], val["t"]
    ec, rc, en, rn = [], [], [], []
    K_s = K_FULL * np.diag([IMG_W / NATIVE_W, IMG_H / NATIVE_H, 1.0])
    for i in range(len(X)):
        f = detector.detect(X[i])
        if ID_M in f:
            Tc, _, _ = marker_pose(f[ID_M], K_s, None, L_M)
            ec.append(np.linalg.norm(Tc[:3, 3] - t[i]) * 1000.0)
            rc.append(rot_error_deg(R[i], Tc[:3, :3]))
        else:
            ec.append(np.nan)
            rc.append(np.nan)
        with torch.no_grad():
            Rn, tn = decode_pose_torch(model(torch.from_numpy(X[i]).unsqueeze(0)))
        en.append(np.linalg.norm(tn[0].numpy() - t[i]) * 1000.0)
        rn.append(rot_error_deg(R[i], Rn[0].numpy()))
    print("\n--- REZULTATE E5 (Sintetic) ---")
    print(f"Clasic | e_t: {stats(ec)} mm | e_R: {stats(rc)} deg | Detectie: {np.isfinite(ec).mean()*100:.1f}%")
    print(f"NN     | e_t: {stats(en)} mm | e_R: {stats(rn)} deg | Detectie: 100.0%")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else ""
    if cmd == "e1":
        exp_e1()
    elif cmd == "e2":
        exp_e2()
    elif cmd == "e3":
        exp_e3()
    elif cmd == "e4":
        exp_e4()
    elif cmd == "e5":
        exp_e5()