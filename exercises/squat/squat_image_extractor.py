# Documentation: https://developers.google.com/mediapipe/solutions/vision/pose_landmarker/python
import time

# Run command in terminal: python -m pip install mediapipe
import cv2 as cv
import mediapipe as mp
from utils.squatClass import Squat


def main(video_path, debug_mode=False):
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

    squat = Squat()
    algorithm_Squat = []

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

            # Make detectionw
            results = pose.process(frame)
            if not results:
                continue

            # Recolor back to BGR
            frame.flags.writeable = True
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)

            # Render key points
            mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            try:
                squat_result = squat.check_squat(results.pose_landmarks)
                algorithm_Squat.append(squat_result)

                if squat_result == 1:
                    display_text = "Squat korrekt ausgeführt"
                    display_color = (0, 255, 0)
                elif squat_result == 2:
                    display_text = "Squat war zu tief"
                    display_color = (0, 0, 255)
                elif squat_result == 3:
                    display_text = "Squat war nicht tief genug"
                    display_color = (0, 0, 255)
                else:
                    display_text = "Squat in Ausführung"
                    display_color = (255, 255, 0)

                cv.putText(frame, display_text, (10, 30), font, 1, display_color, 2, cv.LINE_AA)

            except AttributeError as e:
                print("Error detecting Squat (key points not detected?): " + str(e))
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
            elif key == ord('e'):
                # save image to disk
                video_name = video_path.split("/")[-1].split(".")[0]
                path = "../../resources/images/squat/labeled_images/" + video_name + f"_frame_{current_frame}.jpg"
                cv.imwrite(path, frame)
                print(f"Saving image to {path}")

        cap.release()
        cv.destroyAllWindows()


if __name__ == "__main__":
    # go through each frame in the video
    debug_mode = True

    # set video path
    #video_path = "C:/Users/mitte/PycharmProjects/Studienarbeit/resources/videos/squat/Cutted_Video.mov"
    video_path = "C:/Users/mitte/StudioProjects/Studienarbeit/android/app/src/main/res/raw/test_video_functionalshirt.mp4"
    main(video_path, debug_mode)
