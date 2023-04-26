# Sterreo-Vision
The aim of this project was to find interesting
features and correspondences between the left and right images
using either the CORNERS and NCC algorithms or SIFT
features and descriptors. The results are displayed by connecting
corresponding features with different colored lines to make it
easier to visualize. A program is also developed to estimate the
Fundamental Matrix for each pair using the correspondences
above and RANSAC to eliminate outliers. Additionally, a dense
disparity map is computed using the Fundamental Matrix to
help reduce the search space. The output includes three images:
one image with the vertical disparity component, another image
with the horizontal disparity component, and a third image
representing the disparity vector using color. The direction of the
vector is coded by hue, and the length of the vector is coded by
saturation. For grayscale display, the disparity values are scaled
so that the lowest disparity is 0 and the highest disparity is 255.
The results are discussed and sample images are presented in
the report.

We explored a total of 4 algorithms in this project. They
are stated below.
- Reading the Images.
- Detecting Harris corner.
- Compute normalized cross-correlation and RANSAC.
- Estimating Fundamental Matrix.
- Compute Dense Disparity Map.

![Epipolar Lines](https://github.com/yashmewada9618/Sterreo-Vision/blob/main/Scripts/Epipolarlines.jpg)
![Disparity](https://github.com/yashmewada9618/Sterreo-Vision/blob/main/Scripts/Disparity.png)
