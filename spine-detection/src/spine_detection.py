# Documentation: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
import time

# Run command in terminal: python -m pip install mediapipe
import numpy as np
import cv2 as cv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import scipy.stats as stats


def get_spine_roi(img, pose_landmarks, padding_top=25, padding_bottom=25, padding_left=100, padding_right=25):
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
    x1 = int(avg_hip_x) - padding_left
    x2 = int(avg_shoulder_x) - padding_right
    y1 = int(avg_shoulder_y) - padding_top
    y2 = int(avg_hip_y) - padding_bottom

    # [y1:y2, x1:x2]
    if x1 <= 0:
        return img
    elif y1 <= 0:
        return img
    elif x2 <= x1:
        return img
    elif y2 <= y1:
        return img

    # cropped_img = img[y1:y2, x1:x2]
    cropped_img = img[y1:y2, x1:x2]
    return cropped_img


def filter_frame(img, lower_bound, upper_bound):
    # Convert BGR to HSV
    img_hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)

    # define yellow color range in HSV
    color_range = cv.inRange(img_hsv, lower_bound, upper_bound)

    # filter yellow areas
    img_filtered = img.copy()
    img_filtered[np.where(color_range == 0)] = 0

    return img_filtered


def detect_spine_contours(img):
    """
    Detect spine contours in image
    :param img: image of person, cropped and filtered
    :return: spine contours as numpy array, `None` if no contours found

    Algorithm: applies gaussian blur and canny edge detection to frame. Assumption: spine is the first edge found in
    the frame as seen from the origin of the image (top left)
    """
    # Convert to grayscale
    img_gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    # Apply Gaussian Blur
    img_blur = cv.GaussianBlur(img_gray, (5, 5), 0)

    # Apply Canny Edge Detection
    img_canny = cv.Canny(img_blur, 100, 110)

    # Konturen extrahieren
    contours, hierarchy = cv.findContours(img_canny, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)

    # Kontur ganz links identifizieren
    # Sortiere die Konturen nach x- und y-Koordinate
    contours = sorted(contours, key=lambda x: cv.boundingRect(x)[0] + cv.boundingRect(x)[1])

    # Wähle die Kontur ganz links
    try:
        leftmost_contour = contours[0]
        return leftmost_contour
    except IndexError:
        return None


def is_spine_straight(frame, keypoints, upper_color_range, lower_color_range, show_filtered=False,
                      show_spine_contours=False):
    """
    Check if spine is straight by detecting the spine curve
    :param frame: image with person
    :param keypoints: detected keypoints with HPE
    :param upper_color_range: Upper bound for color range in HSV of t-shirt color
    :param lower_color_range: Lower bound for color range in HSV of t-shirt color
    :param show_filtered: show frame after filtering color range
    :param show_spine_contours: show frame with detected spine contours
    :return: true if spine is straight, false if spine is round
    """
    cropped_frame = get_spine_roi(frame, keypoints)
    # cv.imshow('ROI', cropped_frame)

    # # Filter yellow color
    # filtered_frame = filter_frame(cropped_frame, lower_color_range, upper_color_range)
    # if show_filtered:
    #     cv.imshow('Filtered', filtered_frame)

    # Detect spine contours
    spine_contours = detect_spine_contours(cropped_frame)
    if spine_contours is None:  # if no contours found
        # display_text = "no contours found"
        return

    # draw contours on cropped and filtered frame
    if show_spine_contours:
        cv.drawContours(cropped_frame, [spine_contours], -1, (0, 0, 255), 2)
        cv.imshow('Spine', cropped_frame)

    # ToDo: better algorithm to detect spine curve
    # Option 1: Fit curve to spine contours and check if curve is straight
    # Option 2: Linear Regression for spine contours --> check sum of squared residuals

    x = spine_contours[:, 0, 0]
    y = spine_contours[:, 0, 1]

    # Fit spine curve
    polynom_spine_curve = np.polyfit(x, y, 2)
    try:
        # ToDo: implement check for hollow back (negative polynom) and sway back (positive polynom)
        if abs(polynom_spine_curve[0]) > 0.004:
            return False
        else:
            return True
    except IndexError:
        print("No curve found.")
        return None

    # Linear Regression
    # slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    # # print("r-squared:", r_value ** 2)
    # if r_value ** 2 > 0.7:
    #     return True
    # else:
    #     return False


def main(video_path, lower_color_range, upper_color_range, debug_mode=False):
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    # define font and text
    font = cv.FONT_HERSHEY_SIMPLEX
    display_text = ""
    display_color = (0, 0, 0)

    # used to record the time when we processed last frame
    prev_frame_time = 0
    # used to record the time at which we processed current frame
    new_frame_time = 0

    frame_counter = 0
    fps_sum = 0

    current_frame = 0

    # Webcam öffnen (Standardkamera, normalerweise 0 oder -1)
    cap = cv.VideoCapture(video_path)
    # Überprüfen, ob die Kamera geöffnet wurde
    if not cap.isOpened():
        print("Can't open video (stream end?). Exiting ...")
        exit()

    # Setup mediapipe instance
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            debug_mode and cap.set(cv.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()

            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

            frame_counter += 1
            new_frame_time = time.time()

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
                                     show_filtered=True, show_spine_contours=True):
                    display_text = "straight spine"
                    display_color = (0, 0, 0)
                else:
                    display_text = "round spine"
                    display_color = (0, 0, 255)
            except AttributeError as e:
                print("Error creating ROI (key points not detected?): " + str(e))
                continue

            # calculate fps
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            fps_sum += fps

            cv.namedWindow('Mediapipe Feed', cv.WINDOW_NORMAL)
            cv.putText(frame, display_text, (10, 30), font, 1, display_color, 2, cv.LINE_AA)
            if debug_mode:
                cv.putText(frame, "Frame nr. = " + str(current_frame), (10, 1070), font, 1, (0, 0, 0), 2, cv.LINE_AA)
            else:
                cv.putText(frame, "FPS=" + str(fps), (10, 1040), font, 1, (0, 0, 0), 2, cv.LINE_AA)
                cv.putText(frame, "Avg FPS=" + str(int(fps_sum / frame_counter)), (10, 1070), font, 1, (0, 0, 0), 2,
                           cv.LINE_AA)
            cv.imshow('Mediapipe Feed', frame)
            cv.resizeWindow('Mediapipe Feed', 270, 540)

            # if cv.waitKey(1) & 0xFF == ord('q'):
            #     break

            # Warte auf eine Taste zum Steuern der Frames (wenn 'q' gedrückt wird, beende die Schleife)
            if debug_mode:
                key = cv.waitKey(0) & 0xFF
            else:
                key = cv.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == ord('d'):
                current_frame += 1
            elif key == ord('a'):
                if current_frame > 0:
                    current_frame -= 1
            elif key == ord('s'):
                if current_frame > 10:
                    current_frame -= 10
            elif key == ord('w'):
                current_frame += 10

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # go through each frame in the video
    debug_mode = True

    # set video path
    video_path = "../../resources/videos/test-data/test_video_functionalshirt.mp4"
    # video_path = "../../resources/videos/squat/squat-yellow-positive_540x1080.mp4"
    # video_path = "../resources/videos/squat/squat-yellow-negative_540x1080.mp4"

    # define color range in HSV
    lower_color_range = np.array([0, 0, 0])
    upper_color_range = np.array([255, 255, 255])

    main(video_path, lower_color_range, upper_color_range, debug_mode)
