import cv2

# Webcam öffnen (Standardkamera, normalerweise 0 oder -1)
cap = cv2.VideoCapture(0)

# Überprüfen, ob die Kamera geöffnet wurde
if not cap.isOpened():
    print("Kamera konnte nicht geöffnet werden.")
    exit()

while cap.isOpened():
    # Frame einlesen
    ret, frame = cap.read()

    if not ret:
        print("Fehler beim Lesen des Frames.")
        break

    # anzeigen des Frames
    cv2.imshow('Webcam Feed', frame)

    # Beenden des Loops, wenn irgendeine Taste oder das "x" im Fenster gedrückt wird
    key = cv2.waitKey(1)
    if key != -1 or cv2.getWindowProperty('Webcam Feed', cv2.WND_PROP_VISIBLE) < 1:
        break

# Freigeben der Kamera und Schließen der OpenCV-Fenster
cap.release()
cv2.destroyAllWindows()



