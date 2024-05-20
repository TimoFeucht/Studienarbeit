# Spine Detection

Detects curvature of the spine in images of people from the side.

Detects ROI for the spine based on the hip and shoulder key points from the HPE model used.
In this example the HPE model is the MediaPipe.
Extracts the spine edge with Canny-Edge and contur-extraction from the ROI and fits a polynomial to the spine.
The curvature of the spine is determined by the polynomial.

## Restrictions

Runs currently only on white background images with yellow t-shirts.

## Usage

Run the `spine_detection.py` script.

Use the variables `video_path`, `lower_color_range` and `upper_color_range` to set the video path 
and the color range of the t-shirt in the video in HSV.

`debug_mode` can be set to either `True`  or `False` to show the video frame by frame with `debug_mode = True` or
not `debug_mode = False`.
The debug mode allows you to control the frames with the keyboard.
Press :

- `d` for the next frame,
- `a` for the previous frame,
- `w` for the next 10 frames,
- `s` for the previous 10 frames and
- `q` to quit the video.

```pyhton
    # go through each frame in the video
    debug_mode = True

    # set video path
    video_path = "../../resources/videos/squat/squat-yellow-positive_540x1080.mp4"
    # video_path = "../resources/videos/squat/squat-yellow-negative_540x1080.mp4"

    # define color range in HSV
    lower_color_range = np.array([0, 150, 100])
    upper_color_range = np.array([30, 255, 255])
```