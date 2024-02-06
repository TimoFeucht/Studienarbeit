import cv2 as cv
import os
import re


def video2images(video_path, output_folder, frame_rate=1):
    # Open the video file
    cap = cv.VideoCapture(video_path)
    i = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_rate == 0:
            cv.imwrite(f"{output_folder}/frame_{i}.jpg", frame)
        i += 1
    cap.release()
    cv.destroyAllWindows()
    print(f"Extracted {i} frames from {video_path}")
    return i


def numerische_sortierung(dateiname):
    # Extrahiere die numerische Sequenz aus dem Dateinamen
    numerische_sequenz = re.search(r'\d+', dateiname)
    if numerische_sequenz:
        return int(numerische_sequenz.group())
    return float('inf')  # Rückgabewert für Dateinamen ohne numerische Sequenz


def frames2csv(frames_folder, csv_path):
    """
    write each frame name in column of csv file,
    create a csv file if not exists
    """
    i = 0
    frames = os.listdir(frames_folder)
    frames = sorted(frames, key=numerische_sortierung)
    with open(csv_path, "w") as f:
        for frame in frames:
            f.write(frame + "\n")
            i += 1

    print(f"Written {i} frames to {csv_path}")


def copy_jpg_files(source_folder, destination_folder, param):
    """"
    cpoy jpg files from source_folder to destination_folder with filename starting from param
    e.g. param = 100 -> frame_100.jpg, frame_101.jpg, ...
    """
    frames = os.listdir(source_folder)
    frames = sorted(frames, key=numerische_sortierung)
    i = 0
    for frame in frames:
        if frame.endswith(".jpg"):
            i += 1
            frame_path = os.path.join(source_folder, frame)
            new_frame_path = os.path.join(destination_folder, f"frame_{param + i}.jpg")
            os.rename(frame_path, new_frame_path)
    print(f"Copied {i} frames from {source_folder} to {destination_folder}")


if __name__ == "__main__":
    # ask user if he really wants to run this script, because it will overwrite files
    print("This script will overwrite files in the output folder. Are you sure you want to run this script?")
    answer = input("Type 'yes' to continue: ")

    if answer != "yes":
        print("Exiting...")
        exit()

    video_path = "../../../resources/videos/squat/videoname.mp4"
    output_folder = "../../../resources/images/squat/test-data"

    # video2images(video_path, output_folder)

    csv_path = "../../../resources/images/squat/test-data/test_data.csv"
    # frames2csv(output_folder, csv_path)

    source_folder = "../../../resources/images/squat/squat-yellow-negative_540x1080"
    destination_folder = "../../../resources/images/squat/test-data"
    # copy_jpg_files(source_folder, destination_folder, 191)