Project: img_mixExample

Description:
This simple CUDA kernel example mixes 2 jpeg images. The images can be color images and they are transformed to grayscale.
By default both image grayscale pixel values are added with an equal weight factor of 0.5, i.e. both images are equally weighted.

Usage:
img_mixExample <Name of first image> <Name of second image> <Name of output image>

All of these images must be standard jpg files and have the ending .jpg.
This is the minimum number of parameter to be specified.

Example:
./img_mixExample blueCircle.jpg redBox.jpg redblueMix.jpg

produces exactly the output image as it can be found in the Github repository

Additional parameters:
<Weight of image 1: p1>: Floating point number typically between 0.0 and 1.0. Specifies the pixel grayscale image weight for the
first image to be mixed. If the weight for the second image is not given, it is calculated as p2 = 1.0 - p1

<Weight of image 2: p2>: Floating point number typically between 0.0 and 1.0. Specifies the pixel grayscale image weight for the
second image.

Example
./img_mixExample blueCircle.jpg redBox.jpg redblueMix.jpg 0.1
Weights the first image with a factor of 0.1, the second one with 0.9.

./img_mixExample blueCircle.jpg redBox.jpg redblueMix.jpg 0.8 0.2
Weights the first image with a factor of 0.8, the second one with 0.2.

How to build:
The CMakeLists.txt is part of the repository.
To build, simply
- Create a build directory under the project
- Navigate into it and call cmake ..
- Build the program with make
- The executable will be available in a /exe subdirectory under the project root. Two example images are there in the repository.

Requirements:
Ubuntu 24.04
CMake
libjpeg-dev
CUDA development environment
