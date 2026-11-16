import cv2
import numpy as np

# 1. Incarcam calibrarea camerei
calib = np.load("camera_calib.npz")
K = calib["K"]
dist = calib["dist"]

# 2. Configuram markerii
L_MM = 92.0  # Latura masurata de tine in mm
ID_F = 0      # ID-ul markerului Fix de pe masa
ID_M = 1      # ID-ul markerului Mobil

# Colturi 3D in sistemul local al markerului (centrul la (0,0,0), plan Z=0)
half_l = L_MM / 2.0
obj_points = np.array([
    [-half_l,  half_l, 0.0],
    [ half_l,  half_l, 0.0],
    [ half_l, -half_l, 0.0],
    [-half_l, -half_l, 0.0]
], dtype=np.float32)

# Detector ArUco cu rafinare subpixel (cerinta T2)
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
parameters = cv2.aruco.DetectorParameters()
parameters.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
detector = cv2.aruco.ArucoDetector(dictionary, parameters)

def solve_marker_pose(corners):
    # SOLVEPNP_IPPE_SQUARE cu rafinare Levenberg-Marquardt (cerinta T3)
    success, rvec, tvec = cv2.solvePnP(
        obj_points, corners, K, dist, flags=cv2.SOLVEPNP_IPPE_SQUARE
    )
    if success:
        cv2.solvePnPRefineLM(obj_points, corners, K, dist, rvec, tvec)
    return rvec, tvec

def make_T(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

def rot_to_euler_deg(R):
    # Rulaj ZYX (Roll-Pitch-Yaw)
    sy = np.sqrt(R[0, 0]**2 + R[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0
    return np.degrees([roll, pitch, yaw])

cap = cv2.VideoCapture(1)
print("=== URMARIRE POSTURA RELATIVA ===")
print("Apasa 'q' pentru a iesi.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    corners, ids, _ = detector.detectMarkers(frame)
    c_T_f = None
    c_T_m = None

    if ids is not None:
        cv2.aruco.drawDetectedMarkers(frame, corners, ids)
        for i, marker_id in enumerate(ids.flatten()):
            rvec, tvec = solve_marker_pose(corners[i][0])
            # Desenam axele 3D (lungime axa: jumatate din latura markerului)
            cv2.drawFrameAxes(frame, K, dist, rvec, tvec, half_l, 2)
            
            if marker_id == ID_F:
                c_T_f = make_T(rvec, tvec)
            elif marker_id == ID_M:
                c_T_m = make_T(rvec, tvec)

    # Calculam postura relativa daca ambii markeri sunt vizibili: F_T_M = (C_T_F)^-1 * C_T_M
    if c_T_f is not None and c_T_m is not None:
        f_T_m = np.linalg.inv(c_T_f) @ c_T_m
        
        tx, ty, tz = f_T_m[:3, 3]
        roll, pitch, yaw = rot_to_euler_deg(f_T_m[:3, :3])
        
        info1 = f"t_rel [mm]: X={tx:6.1f}  Y={ty:6.1f}  Z={tz:6.1f}"
        info2 = f"rot [deg]:  R={roll:5.1f}  P={pitch:5.1f}  Y={yaw:5.1f}"
        cv2.putText(frame, info1, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, info2, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "Astept ambii markeri in cadru...", (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    cv2.imshow("Tracking Relativ F -> M (T3)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()