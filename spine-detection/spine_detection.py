# Documentation: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python

# Run command in terminal: python -m pip install mediapipe
import numpy as np
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision


def get_roi(img, pose_landmarks, padding_top=100, padding_bottom=25, padding_left=100, padding_right=75):
    mp_pose = mp.solutions.pose
    image_height, image_width, _ = img.shape

    left_shoulder_x = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].x * image_width
    left_shoulder_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height

    right_shoulder_x = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].x * image_width
    right_shoulder_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height

    left_hip_x = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].x * image_width
    left_hip_y = pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y * image_height

    right_hip_x = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].x * image_width
    right_hip_y = pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y * image_height

    avg_shoulder_x = (left_shoulder_x + right_shoulder_x) / 2
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    avg_hip_x = (left_hip_x + right_hip_x) / 2
    avg_hip_y = (left_hip_y + right_hip_y) / 2

    # crop image
    cropped_img = img[int(avg_shoulder_y) - padding_top: int(avg_hip_y) - padding_bottom,
                      int(avg_hip_x) - padding_left: int(avg_shoulder_x) - padding_right]
    return cropped_img


def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # Webcam öffnen (Standardkamera, normalerweise 0 oder -1)
    cap = cv2.VideoCapture(0)
    # Überprüfen, ob die Kamera geöffnet wurde
    if not cap.isOpened():
        print("Kamera konnte nicht geöffnet werden.")
        exit()

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Fehler beim Lesen des Frames.")
                break

            # Recolor image to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame.flags.writeable = False

            # Make detection
            results = pose.process(frame)

            # Recolor back to BGR
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Render detections
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2)
                                      )

            # if results.pose_landmarks:
            #     cropped_frame = get_roi(frame, results.pose_landmarks)
            #     frame = cropped_frame

            cv2.imshow('Mediapipe Feed', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
