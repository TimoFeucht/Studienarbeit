from .exerciseUtils import extract_keypoint, calculate_angle_between_three_points
import mediapipe as mp


def is_l_sit(person) -> int:
    THRESHOLD_HIP = 96
    THRESHOLD_MIN = 172
    THRESHOLD_PERFECT = 175
    THRESHOLD_MAX = 183

    mp_pose = mp.solutions.pose

    left_knee = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_KNEE)
    right_knee = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_KNEE)
    left_hip = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_HIP)
    right_hip = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_HIP)
    left_ankle = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_ANKLE)
    right_ankle = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_ANKLE)
    left_shoulder = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_SHOULDER)
    right_shoulder = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_SHOULDER)
    left_elbow = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_ELBOW)
    right_elbow = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_ELBOW)
    left_wrist = extract_keypoint(person, mp_pose.PoseLandmark.LEFT_WRIST)
    right_wrist = extract_keypoint(person, mp_pose.PoseLandmark.RIGHT_WRIST)

    angle_left_knee = calculate_angle_between_three_points(left_hip, left_knee, left_ankle)
    angle_right_knee = calculate_angle_between_three_points(right_hip, right_knee, right_ankle)
    # angle_left_elbow = calculate_angle_between_three_points(left_shoulder, left_elbow, left_wrist)
    # angle_right_elbow = calculate_angle_between_three_points(right_shoulder, right_elbow, right_wrist)
    angle_left_hip = calculate_angle_between_three_points(left_shoulder, left_hip, left_knee)
    angle_right_hip = calculate_angle_between_three_points(right_shoulder, right_hip, right_knee)

    # print("Angle left Hip: ", angle_left_hip)
    # print("Angle right Hip: ", angle_right_hip)
    # print("Angle left Knee: ", angle_left_knee)
    # print("Angle right Knee: ", angle_right_knee)
    # print("Angle left Elbow: ", angle_left_elbow)
    # print("Angle right Elbow: ", angle_right_elbow)

    if angle_left_hip < THRESHOLD_HIP or angle_right_hip < THRESHOLD_HIP:
        if ((THRESHOLD_PERFECT < angle_left_knee < THRESHOLD_MAX) or
                (THRESHOLD_PERFECT < angle_right_knee < THRESHOLD_MAX)):
            return 2
        elif ((THRESHOLD_MIN < angle_left_knee < THRESHOLD_MAX) or
              (THRESHOLD_MIN < angle_right_knee < THRESHOLD_MAX)):
            return 1

    return 0
