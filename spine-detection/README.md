# Spine Detection
Detects curvature of the spine in images of people from the side.

Detects ROI for the spine based on the hip and shoulder key points from the HPE model used.
In this example the HPE model is the MediaPipe.
Extracts the spine edge with Canny-Edge and contur-extraction from the ROI and fits a polynomial to the spine.
The curvature of the spine is determined by the polynomial.


## Restrictions
Runs currently only on white background images with yellow t-shirts.