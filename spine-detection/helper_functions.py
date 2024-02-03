import cv2 as cv


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


if __name__ == "__main__":
    video_path = "../resources/videos/squat/squat-yellow-positive_540x1080.mp4"
    output_folder = "../resources/images/squat/squat-yellow-positive_540x1080"

    video2images(video_path, output_folder)
