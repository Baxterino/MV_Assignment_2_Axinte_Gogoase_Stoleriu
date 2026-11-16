import cv2
import numpy as np

# Configurare tablă: numărul de colțuri INTERIOARE (unde se intersectează 4 pătrate)
# Pentru o tablă standard 10x7 pătrate, colțurile interioare sunt 9x6:
CHECKERBOARD = (9, 6)
# Dimensiunea laturii unui pătrat în milimetri (ex. 25.0 mm sau măsoară cu rigla)
SQUARE_SIZE_MM = 21.0

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Coordonatele 3D ale colțurilor pe tabla plană (Z=0)
objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_MM

objpoints = []  # Puncte 3D în coordonatele lumii reale
imgpoints = []  # Puncte 2D în planul imaginii

cap = cv2.VideoCapture(1)
captured_frames = 0

print("=== CALIBRARE WEBCAM ===")
print("Instructiuni:")
print(" - Tine tabla de sah in fata camerei.")
print(" - Cand colturile sunt detectate (colorate), apasa SPACE pentru a salva cadrul.")
print(" - Captureaza cel putin 12-15 cadre variind distanta si inclinarea (tilt).")
print(" - Apasa C cand ai suficiente cadre pentru a rula calibrarea.")
print(" - Apasa Q pentru a iesi.")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret_corners, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, None)

    display = frame.copy()

    if ret_corners:
        cv2.drawChessboardCorners(display, CHECKERBOARD, corners, ret_corners)
        cv2.putText(
            display,
            "Colturi gasite! Apasa SPACE",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2,
        )
    else:
        cv2.putText(
            display,
            "Nu se vad toate colturile",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
        )

    cv2.putText(
        display,
        f"Cadre salvate: {captured_frames}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 0),
        2,
    )
    cv2.imshow("Calibrare Camera", display)

    key = cv2.waitKey(1) & 0xFF
    if key == ord(" ") and ret_corners:
        corners2 = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), criteria
        )
        objpoints.append(objp)
        imgpoints.append(corners2)
        captured_frames += 1
        print(f"Cadru capturat: {captured_frames}")
    elif key == ord("c") or key == ord("C"):
        if captured_frames >= 10:
            break
        else:
            print("Ai nevoie de cel putin 10 cadre pentru o calibrare valida!")
    elif key == ord("q") or key == ord("Q"):
        cap.release()
        cv2.destroyAllWindows()
        exit(0)

cap.release()
cv2.destroyAllWindows()

print("\nCalculare parametri intrinseci (K, dist)...")
ret_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
    objpoints, imgpoints, gray.shape[::-1], None, None
)

print("-" * 50)
print(f"Calibrare incheiata cu succes!")
print(f"RMS Re-projection Error: {ret_rms:.4f} pixeli (ideal < 0.5 - 1.0)")
print("Matricea K:")
print(K)
print(f"Coeficienti de distorsiune: {dist.ravel()}")
print("-" * 50)

# Salvare fisier
np.savez("camera_calib.npz", K=K, dist=dist, rms=ret_rms)
print("Salvat in 'camera_calib.npz'!")