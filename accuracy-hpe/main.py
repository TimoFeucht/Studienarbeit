import glob
import pandas as pd
import cv2 as cv
import mediapipe as mp
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score


# Lese die gelabelten Daten ein
json_path = ("../../resources/images/labeled_images/Label_Tensorflow_And_Hand_Labeling/Studienarbeit_Labeling_HPE_v1"
             ".json")

# Image-path
img_path = glob.glob("../../resources/images/labeled_images/Label_Tensorflow_And_Hand_Labeling/all/*.jpg")

# load all images from the folder
images = []
for i in img_path:
    img = cv.imread(i)
    images.append(img)


# Results from the algorithm
algorithm_LSit = []

print("Starting L-Sit detection and collecting results...")

# Setup mediapipe instance
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Setup mediapipe instance
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    for img in images:
        # Recolor image to RGB
        img_keypoints = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img_keypoints.flags.writeable = False

        # Make detection
        results = pose.process(img_keypoints)
        if not results:
            continue

        # Recolor back to BGR
        img_keypoints.flags.writeable = True
        img_keypoints = cv.cvtColor(img_keypoints, cv.COLOR_RGB2BGR)

        # Render key points
        mp_drawing.draw_landmarks(img_keypoints, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                  mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                  mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

        # show frame
        cv.imshow('LSit Detection', img_keypoints)

        try:
            print("Calculating ...")
        except AttributeError as e:
            print("Error detecting (key points not detected?): " + str(e))
            continue

        # Display frame
        # cv.imshow('Spine Detection', frame)
