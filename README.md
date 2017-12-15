# Pose Estimation and Optical Flow
** Computer Vision Meets Machine Learning, Fall 2017**

Mohammad Afshar, ma2510@nyu.edu

[Radhika Mattoo](https://github.com/radhikamattoo), rm3485@nyu.edu

## Overview

This project uses Keras' model.h5 model to perform pose estimation on a series of images we created.

We then used the first frame of a video we created from the series of images to extract key points on the skeleton, and used these points as input into Lucas-Kanade optical flow to see if the program would be able to track the skeleton during the video.

The rest of this README details our process/results, as well as details on how to run and build our program.

## Results

1. Pose Estimation

    The thing we performed was pose estimation on a series of images we took of Mohammad, where he slowly moved across the screen. Below are a before and after images for 1 frame.

    Original Image:

    ![Original](./data_archive/init_data/figure_1_reg.png)

    Pose Estimation Results:

    ![Skeleton](./data_archive/pose_results/figure_1_skeleton.png)

    The results from the Keras model are quite good, despite the drastic change in lighting, and the fact that the person is not visible below the knees.

2. Optical Flow

    We decided to use the Lucas-Kanade algorithm offered by OpenCV to perform optical flow. We wanted to run the algorithm on the **original** images, to see if it would be able to track the skeleton.

    However, how would the algorithm know to track the skeleton points on the original image? The built-in OpenCV function `goodFeaturesToTrack` chooses arbitrary points, but we wanted the algorithm to specifically track points on the skeleton found by the Keras model. Thus, we had to manually extract the skeleton points on the original image by comparing the points to the corresponding skeleton image.

    Once we had extracted the skeleton points on the first original image, we then fed them into the Optical Flow algorithm.

    Unfortunately, our images had too much variance in position, so the optical flow algorithm couldn't track the points correctly. Below is a sample of our results: 

    ![Wrong Result](./data_archive/op_flow_results/1.png)



## Running
