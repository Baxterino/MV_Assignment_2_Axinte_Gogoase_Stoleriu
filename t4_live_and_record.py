import time
import cv2
import matplotlib.pyplot as plt
import numpy as np
from animate_relative_pose import animate, rpy_deg
from mvlib import MarkerDetector, marker_pose, relative_pose

# --- Configurare hardware & markeri ---
CALIB_FILE = "camera_calib.npz"
L_M = 0.092  # Latura markerului in METRI (205 mm)
ID_F = 0  # Markerul Fix (F)
ID_M = 1  # Markerul Mobil (M)
RECORD_SECONDS = 20.0

calib = np.load(CALIB_FILE)
K, dist = calib["K"], calib["dist"]
detector = MarkerDetector(cv2.aruco.DICT_6X6_250, subpixel=True)

# --- Initializare fereastra Matplotlib LIVE (3D + Grafice in timp) ---
plt.ion()
fig = plt.figure(figsize=(10, 5))
ax3 = fig.add_subplot(1, 2, 1, projection="3d")
axp = fig.add_subplot(2, 2, 2)
axr = fig.add_subplot(2, 2, 4)

# --- Deschidere webcam ---
cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
if not cap.isOpened():
  cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print("=== T4: LIVE TRACKING & RECORDING ===")
print("Taste in fereastra OpenCV:")
print("  [R] - Porneste / Opreste inregistrarea de 20 secunde")
print("  [C] - Sterge traiectoria curenta din grafic")
print("  [Q] - Iesire")

live_t, live_p, live_e = [], [], []
rec_t, rec_poses, rec_frames = [], [], []
recording = False
t0_live = time.time()
t0_rec = None
last_T_CF = None
frame_idx = 0

while True:
  ret, frame = cap.read()
  if not ret:
    break

  now = time.time()
  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
  found = detector.detect(gray)

  # Desenam markeri si axe pe imaginea camerei
  if ID_F in found:
    T_CF, _, _ = marker_pose(found[ID_F], K, dist, L_M)
    last_T_CF = T_CF
    rvec_f, _ = cv2.Rodrigues(T_CF[:3, :3])
    cv2.drawFrameAxes(frame, K, dist, rvec_f, T_CF[:3, 3], L_M * 0.5, 2)
  else:
    # Daca F e acoperit scurt timp si camera sta pe loc, pastram ultima postura F
    T_CF = last_T_CF

  T_FM = None
  if T_CF is not None and ID_M in found:
    T_CM, _, _ = marker_pose(found[ID_M], K, dist, L_M)
    rvec_m, _ = cv2.Rodrigues(T_CM[:3, :3])
    cv2.drawFrameAxes(frame, K, dist, rvec_m, T_CM[:3, 3], L_M * 0.5, 2)

    T_FM = relative_pose(T_CF, T_CM)
    p = T_FM[:3, 3]
    e = rpy_deg(T_FM[:3, :3])

    # Actualizare buffer live (pastram ultimele 150 de puncte)
    live_t.append(now - t0_live)
    live_p.append(p)
    live_e.append(e)
    if len(live_t) > 150:
      live_t.pop(0)
      live_p.pop(0)
      live_e.pop(0)

    cv2.putText(
        frame,
        f"XYZ [mm]: {p[0]*1000:.1f}, {p[1]*1000:.1f}, {p[2]*1000:.1f}",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
    cv2.putText(
        frame,
        f"RPY [deg]: {e[0]:.1f}, {e[1]:.1f}, {e[2]:.1f}",
        (15, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 0),
        2,
    )
  else:
    cv2.putText(
        frame,
        "Astept markeri F si M...",
        (15, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )

  # --- Logica de inregistrare 20s ---
  if recording:
    elapsed = now - t0_rec
    cv2.circle(frame, (frame.shape[1] - 30, 30), 10, (0, 0, 255), -1)
    cv2.putText(
        frame,
        f"REC: {elapsed:.1f}s / {RECORD_SECONDS:.0f}s",
        (frame.shape[1] - 210, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 0, 255),
        2,
    )

    if T_FM is not None:
      rec_t.append(elapsed)
      rec_poses.append(T_FM.copy())
      rec_frames.append(frame.copy())

    if elapsed >= RECORD_SECONDS:
      recording = False
      print("\nInregistrare de 20s completa! Generare exporturi...")
      break

  # --- Actualizare grafic LIVE la fiecare 3 cadre (pentru FPS bun) ---
  frame_idx += 1
  if frame_idx % 3 == 0 and len(live_p) > 1:
    P_arr = np.array(live_p)
    E_arr = np.array(live_e)
    T_arr = np.array(live_t)

    ax3.cla()
    ax3.set_title("Traiectorie 3D in sistemul F")
    ax3.set(xlim=(-0.5, 0.5), ylim=(-0.5, 0.5), zlim=(-0.1, 0.5))
    ax3.set_xlabel("X [m]")
    ax3.set_ylabel("Y [m]")
    ax3.set_zlabel("Z [m]")

    # Desenam axele F (origine) si M (pozitia curenta)
    cur_T = np.eye(4)
    cur_T[:3, :3] = T_FM[:3, :3] if T_FM is not None else np.eye(3)
    cur_T[:3, 3] = P_arr[-1]
    for T_mat, scale in ((np.eye(4), 0.12), (cur_T, 0.08)):
      for i, c in enumerate("rgb"):
        ax3.plot(
            *zip(T_mat[:3, 3], T_mat[:3, 3] + scale * T_mat[:3, i]),
            color=c,
            lw=2,
        )
    ax3.plot(P_arr[:, 0], P_arr[:, 1], P_arr[:, 2], "k:", lw=1.2)

    axp.cla()
    for i, lbl in enumerate(("x", "y", "z")):
      axp.plot(T_arr, P_arr[:, i] * 1000.0, label=lbl)
    axp.set_ylabel("Pozitie [mm]")
    axp.legend(loc="upper right", ncol=3, fontsize=8)
    axp.grid(True, alpha=0.3)

    axr.cla()
    for i, lbl in enumerate(("roll", "pitch", "yaw")):
      axr.plot(T_arr, E_arr[:, i], label=lbl)
    axr.set_ylabel("Unghi [deg]")
    axr.set_xlabel("Timp [s]")
    axr.legend(loc="upper right", ncol=3, fontsize=8)
    axr.grid(True, alpha=0.3)

    plt.pause(0.001)

  cv2.imshow("Webcam Tracking (T4)", frame)
  key = cv2.waitKey(1) & 0xFF
  if key == ord("q"):
    break
  elif key == ord("c"):
    live_t.clear()
    live_p.clear()
    live_e.clear()
  elif key == ord("r") and not recording:
    print("Pornit inregistrare 20s! Misca markerul M...")
    rec_t.clear()
    rec_poses.clear()
    rec_frames.clear()
    t0_rec = time.time()
    recording = True

cap.release()
cv2.destroyAllWindows()
plt.ioff()
plt.close(fig)

# --- Export sincronizat la finalul inregistrarii ---
if len(rec_poses) > 5:
  # 1. Salvam video-ul camerei (MP4) la FPS-ul real al inregistrarii
  fps_real = len(rec_frames) / max(rec_t[-1], 1e-3)
  h, w = rec_frames[0].shape[:2]
  out_vid = cv2.VideoWriter(
      "t4_camera_video.mp4", cv2.VideoWriter_fourcc(*"mp4v"), fps_real, (w, h)
  )
  for f in rec_frames:
    out_vid.write(f)
  out_vid.release()
  print(f"Salvat 't4_camera_video.mp4' ({fps_real:.1f} FPS)")

  # 2. Salvam datele brute (.npz) ca sa le ai si pentru grafice/E3
  np.savez(
      "t4_recorded_seq.npz",
      times=np.array(rec_t),
      poses=np.array(rec_poses),
  )
  print("Salvat 't4_recorded_seq.npz'")

  # 3. Exportam animatia 3D sincronizata folosind functia din animate_relative_pose.py
  print("Se genereaza animatia GIF (dureaza cateva secunde)...")
  # Sub-esantionam la ~15 fps daca sunt foarte multe cadre pentru viteza la export GIF
  step = max(1, int(round(fps_real / 15.0)))
  animate(
      np.array(rec_t)[::step], rec_poses[::step], out="t4_relative_pose.gif"
  )
  print("Salvat 't4_relative_pose.gif'! Task T4 complet.")