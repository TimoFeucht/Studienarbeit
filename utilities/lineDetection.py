import cv2
import numpy as np


# Webcam öffnen (Standardkamera, normalerweise 0 oder -1)
cap = cv2.VideoCapture(0)

# Überprüfen, ob die Kamera geöffnet wurde
if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden.")
    exit()

while cap.isOpened():
    # Frame einlesen
    ret, orig_frame = cap.read()
    if not ret:
        print("Fehler beim Lesen des Frames.")
        break

    # frame = cv2.GaussianBlur(orig_frame, (5, 5), 0)
    frame = orig_frame
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower_bound = np.array([0, 0, 150])
    upper_bound = np.array([180, 180, 180])

    mask = cv2.inRange(hsv, lower_bound, upper_bound)
    edges = cv2.Canny(mask, 75, 150)

    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 50, maxLineGap=50)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(frame, (x1, y1), (x2, y2), (0, 255, 0), 5)

    cv2.imshow("frame", frame)
    cv2.imshow("edges", edges)
    cv2.imshow("mask", mask)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Freigeben der Kamera und Schließen der OpenCV-Fenster
cap.release()
cv2.destroyAllWindows()