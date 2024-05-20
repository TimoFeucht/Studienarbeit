import pandas as pd
import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

from spine_detection import is_spine_straight

# Lese die gelabelten Daten ein
csv_path = "../../resources/videos/test-data/test_video_functionalshirt.csv"
labels_df = pd.read_csv(csv_path, delimiter=";", header=None)

# Video-Datei
video_path = "../../resources/videos/test-data/test_video_functionalshirt.mp4"

# Farbfilter für die Hautfarbe
lower_color_range = np.array([0, 0, 0])
upper_color_range = np.array([255, 255, 255])

prev_spine_detection = False

# Results from the algorithm
algorithm_spine_results = []  # Hier 1 für gerade, 0 für nicht gerade

print("Starting spine detection and collecting results...")

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

        try:
            # check if spine is straight
            if is_spine_straight(frame, results.pose_landmarks, upper_color_range, lower_color_range,
                                 previous_frame=prev_spine_detection, show_filtered=False, show_spine_contours=False):
                prev_spine_detection = True
                algorithm_spine_results.append(1)
            else:
                prev_spine_detection = False
                algorithm_spine_results.append(0)
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
actual_labels = labels_df[1].tolist()
# ectract file name of true positive, false positive, true negative and false negative
# true positive
true_positive = [i for i in range(len(actual_labels)) if actual_labels[i] == 1 and algorithm_spine_results[i] == 1]
# false positive
false_positive = [i for i in range(len(actual_labels)) if actual_labels[i] == 0 and algorithm_spine_results[i] == 1]
# true negative
true_negative = [i for i in range(len(actual_labels)) if actual_labels[i] == 0 and algorithm_spine_results[i] == 0]
# false negative
false_negative = [i for i in range(len(actual_labels)) if actual_labels[i] == 1 and algorithm_spine_results[i] == 0]

print("True Positive:", true_positive)
print("False Positive:", false_positive)
print("True Negative:", true_negative)
print("False Negative:", false_negative)

# Berechne die Confusion Matrix und Metriken
conf_matrix = confusion_matrix(actual_labels, algorithm_spine_results)
accuracy = accuracy_score(actual_labels, algorithm_spine_results)
precision = precision_score(actual_labels, algorithm_spine_results)
recall = recall_score(actual_labels, algorithm_spine_results)
# f score
f1 = f1_score(actual_labels, algorithm_spine_results)

print("Confusion Matrix:\n", conf_matrix)
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1 Score:", f1)

# Best results so far:

## 0.005 polynom fitting:
# Confusion Matrix:
#  [[347  50]
#  [ 72  99]]
# Accuracy: 0.7852112676056338
# Precision: 0.6644295302013423
# Recall: 0.5789473684210527
# F1 Score: 0.61875

## 0.004 polynom fitting:
# Confusion Matrix:
#  [[365  32]
#  [ 92  79]]
# Accuracy: 0.7816901408450704
# Precision: 0.7117117117117117
# Recall: 0.4619883040935672
# F1 Score: 0.5602836879432624