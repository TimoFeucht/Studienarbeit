from exercises.lsit.utils.exerciseUtils import extract_keypoint, calculate_angle_between_three_points
import mediapipe as mp

class Squat:
    KNEE_ANGLE_THRESHOLD_MIN = 75
    KNEE_ANGLE_THRESHOLD_MAX = 95

    def __init__(self):
        self.previous_knee_angle = float('inf')

    def check_squat(self, person) -> int:
        current_knee_angle = self.calculate_average_knee_angle(person)

        if current_knee_angle < self.KNEE_ANGLE_THRESHOLD_MIN:
            return 2  # Squat too deep
        elif current_knee_angle > self.KNEE_ANGLE_THRESHOLD_MAX:
            return 3  # Squat not deep enough
        else:
            return 1  # Correct squat

    def calculate_average_knee_angle(self, person):
        mp_pose = mp.solutions.pose
        left_knee = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_KNEE)
        right_knee = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_KNEE)
        left_hip = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_HIP)
        right_hip = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_HIP)
        left_ankle = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_ANKLE)
        right_ankle = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_ANKLE)

        knee_left = calculate_angle_between_three_points(left_hip, left_knee, left_ankle)
        knee_right = calculate_angle_between_three_points(right_hip, right_knee, right_ankle)
        print(knee_left, knee_right)
        return (knee_left + knee_right) / 2
