#Advanced Lane Finding Project

Usage:
```
python srcips/run.py path/to/video_file
```

##1. Goals
The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/undistort_output.png "Undistorted"
[image2]: ./examples/raw_to_undistorted.png "Raw to Undistorted"
[image3]: ./examples/threshold_image.png "Threshold Lane Image"
[image4]: ./examples/perspective_transform.png "Perspective Transform"
[image5]: ./examples/lane_detection.png "Lane Detection"
[image6]: ./examples/final_image.png "Output"


##2. [Rubric Points](https://review.udacity.com/#!/rubrics/571/view) 
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in file scripts/calibrate_camera.py in function calibrate.
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objpoints` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I save this values to load and apply them later much faster. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

![Example of undistorted Image of check-board.][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.
To do this step, I apply the distortion correction using the calculated camera matrix and distortion coefficients. As a result incoming distorted camera image become undistorted:
![Raw to Undistorted][image2]

####2. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform is in scripts/roi_utils.py includes a function called `transform()`,  The `transform()` function takes as inputs an image (`img`), as well as source (`src_points`) and destination (`dst_points`) points.  I chose the hardcode the source and destination points in the following manner:

```
roishape = np.array([[[0, Y-30], 
                      [int(X/2)-100, Y-int(Y/2.7)], 
                      [int(X/2)+100, Y-int(Y/2.7)], 
                      [X,Y-30]]], 
                      dtype=np.int32)

roi_dst_shape = np.array([[[0, Y-30], 
                           [0, 0], 
                           [X, 0], 
                           [X,Y-30]]], 
                           dtype=np.int32)

```
Where X and Y are the shape of the source video frames.
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
|   0,  690      | 0, 690       | 
| 540,  454      | 0, 0         |
| 740,  454      | 1280, 0      |
| 1280, 690      | 1280, 690    |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![Perspective transform.][image4]

####3. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (threshold function in scripts/roi_utils.py).  Here's an example of my output for this step.

![Threshold of lane image.][image3]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

In scripts/lane.py is a detect function which detects the lane from histogram on first frame and by previous borders for all other frames.
To identify the lane position on first frame I used histogram peeks, which works well if there is only low noise level on the threshold image.
On all the frames after first one I searched in the area of the last one (see following image). 

![Lane Detection.][image5]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature is calculated in `scripts/lane.py` in `curvature()` function. And displacement of the car on the lane is calculated within `scripts/utils.py` in `draw_lane()` function (should be actually an extra function). To translate pixel values into meters I used constants for US proposed by Udacity.

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in `scripts/utils.py` in `draw_lane()` and `draw_curve()` functions.  Here is an example of my result on a test image:

![Final image after lane detection on it.][image6]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](https://youtu.be/0zbGppQ42w0)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
The pipeline works well for straight lanes with good visible border lines. But if the road is noisy and has lot of sharp curves the pipeline is not well enough. To approach this problems it would make sense to apply advanced threshold and detection techniques maybe with additional checks to increase the confidence.
