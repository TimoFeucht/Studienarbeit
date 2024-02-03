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


if __name__ == "__main__":
    video_path = "../../../resources/videos/squat/squat-yellow-positive_540x1080.mp4"
    output_folder = "../../../resources/images/squat/squat-yellow-positive_540x1080"

    # video2images(video_path, output_folder)

    csv_path = "../../../resources/images/squat/squat-yellow-positive_540x1080/squat-yellow-positive_540x1080.csv"
    # frames2csv(output_folder, csv_path)
