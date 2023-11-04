import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import cv2

# Import matplotlib libraries
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# import helper functions
from tensorflow_helper_functions_for_visualization import draw_prediction_on_image

# https://www.tensorflow.org/hub/tutorials/movenet


# "tflite_movenet_thunder_f16" "tflite_movenet_lightning_f16" "tflite_movenet_lightning_int8" "tflite_movenet_thunder_int8" "movenet_lightning" "movenet_thunder"
model_name = "tflite_movenet_thunder_int8"
model_path = ""
input_size = 0

# Load Model from TF hub or TFLite
if "tflite" in model_name:
    if "movenet_lightning_f16" in model_name:
        # !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/float16/4?lite-format=tflite
        model_path = r'tensorflow_models\lite-model_movenet_singlepose_lightning_tflite_float16_4.tflite'
        input_size = 192
    elif "movenet_thunder_f16" in model_name:
        # !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/float16/4?lite-format=tflite
        model_path = r'tensorflow_models\lite-model_movenet_singlepose_thunder_tflite_float16_4.tflite'
        input_size = 256
    elif "movenet_lightning_int8" in model_name:
        # !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/lightning/tflite/int8/4?lite-format=tflite
        model_path = r'tensorflow_models\lite-model_movenet_singlepose_lightning_tflite_int8_4.tflite'
        input_size = 192
    elif "movenet_thunder_int8" in model_name:
        # !wget -q -O model.tflite https://tfhub.dev/google/lite-model/movenet/singlepose/thunder/tflite/int8/4?lite-format=tflite
        model_path = r'tensorflow_models\lite-model_movenet_singlepose_thunder_tflite_int8_4.tflite'
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)

    # Initialize the TFLite interpreter
    interpreter = tf.lite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()


    def movenet(input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        # TF Lite format expects tensor type of uint8.
        input_image = tf.cast(input_image, dtype=tf.uint8)
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        interpreter.set_tensor(input_details[0]['index'], input_image.numpy())
        # Invoke inference.
        interpreter.invoke()
        # Get the model prediction.
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])
        return keypoints_with_scores

else:
    if "movenet_lightning" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/lightning/4")
        input_size = 192
    elif "movenet_thunder" in model_name:
        module = hub.load("https://tfhub.dev/google/movenet/singlepose/thunder/4")
        input_size = 256
    else:
        raise ValueError("Unsupported model name: %s" % model_name)


    def movenet(input_image):
        """Runs detection on an input image.

        Args:
          input_image: A [1, height, width, 3] tensor represents the input image
            pixels. Note that the height/width should already be resized and match the
            expected input resolution of the model before passing into this function.

        Returns:
          A [1, 1, 17, 3] float numpy array representing the predicted keypoint
          coordinates and scores.
        """
        model = module.signatures['serving_default']

        # SavedModel format expects tensor type of int32.
        input_image = tf.cast(input_image, dtype=tf.int32)
        # Run model inference.
        outputs = model(input_image)
        # Output is a [1, 1, 17, 3] tensor.
        keypoints_with_scores = outputs['output_0'].numpy()
        return keypoints_with_scores

# Run Inference on Webcam Feed
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera could not be opened.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Get timestamp for pose landmarker task
    frame_timestamp_ms = cap.get(cv2.CAP_PROP_POS_MSEC)

    # Display webcam feed without pose estimation
    # cv2.imshow('Webcam Feed', frame)

    # Resize and pad the image to keep the aspect ratio and fit the expected size.
    input_image = tf.expand_dims(frame, axis=0)
    input_image = tf.image.resize_with_pad(input_image, input_size, input_size)

    # Run model inference.
    keypoints_with_scores = movenet(input_image)

    # Visualize the predictions with image.
    display_image = tf.expand_dims(frame, axis=0)
    display_image = tf.cast(tf.image.resize_with_pad(
        display_image, 640, 640), dtype=tf.int32)
    output_overlay = draw_prediction_on_image(
        np.squeeze(display_image.numpy(), axis=0), keypoints_with_scores)

    # Show the frame
    cv2.imshow('MoveNet Pose Estimation', output_overlay)

    # Press 'q' to quit
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
