import pandas as pd
import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from utils.lsitCalculation import is_l_sit

# Lese die gelabelten Daten ein
csv_path = "../resources/videos/lsit/test_video_lsit_1.csv"
labels_df = pd.read_csv(csv_path, delimiter=";", header=None)

# Video-Datei
video_path = "../resources/videos/lsit/test_video_lsit_1.MP4"


# Results from the algorithm
algorithm_LSit = []

print("Starting L-Sit detection and collecting results...")

# Setup mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Webcam öffnen (Standardkamera, normalerweise 0 oder -1)
cap = cv.VideoCapture(video_path)
# Überprüfen, ob die Kamera geöffnet wurde
if not cap.isOpened():
    print("Can't open video (stream end?). Exiting ...")
    exit()

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        # Recolor image to RGB
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        frame.flags.writeable = False

        # Make detection
        results = pose.process(frame)
        if not results:
            continue

        # Recolor back to BGR
        frame.flags.writeable = True
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

        # Render key points
        # mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        #                           mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
        #                           mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # show frame
        cv.imshow('LSit Detection', frame)

        try:
            lsit = is_l_sit(results.pose_landmarks)
            # check if spine is straight
            if lsit == 1:
                algorithm_LSit.append(1)
                # print("LSit detected!")
            elif lsit == 2:
                algorithm_LSit.append(2)
                # print("Perfect LSit detected!")
            else:
                algorithm_LSit.append(0)
                # print("LSit not detected!")
        except AttributeError as e:
            print("Error detecting spine (key points not detected?): " + str(e))
            continue

        # Display frame
        # cv.imshow('Spine Detection', frame)

        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv.destroyAllWindows()

print("Spine detection finished, calculating confusion matrix and metrics...")

# Vergleiche mit den tatsächlichen Labels
print(labels_df[1].unique())
actual_labels = labels_df[1].tolist()

# Berechne die Confusion Matrix und Metriken
conf_matrix = confusion_matrix(actual_labels, algorithm_LSit)
accuracy = accuracy_score(actual_labels, algorithm_LSit)
precision = precision_score(actual_labels, algorithm_LSit, average='macro')
recall = recall_score(actual_labels, algorithm_LSit, average='macro')
f1 = f1_score(actual_labels, algorithm_LSit, average='macro')

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
