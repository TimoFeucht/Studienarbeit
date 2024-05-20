import math
import mediapipe as mp


class PointF:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def calculate_angle_between_three_points(a, b, c):
    ba = PointF(a.x - b.x, a.y - b.y)
    bc = PointF(c.x - b.x, c.y - b.y)

    dot_product = ba.x * bc.x + ba.y * bc.y

    magnitude_ba = math.sqrt(ba.x ** 2 + ba.y ** 2)
    magnitude_bc = math.sqrt(bc.x ** 2 + bc.y ** 2)

    return math.degrees(math.acos(dot_product / (magnitude_ba * magnitude_bc)))


def extract_keypoint(pose_landmarks, landmark):
    return PointF(pose_landmarks.landmark[landmark].x, pose_landmarks.landmark[landmark].y)