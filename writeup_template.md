


**Advanced Lane Finding Project**

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

[image1]: ./calibration3.jpg "Undistorted"
[image1]: ./camera_cal/calibration3.jpg "Undistorted"
[image2]: ./output_images/straight_lines1_combined.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Road Transformed"
[image3]: ./examples/binary_combo_example.jpg "Binary Example"
[image4]: ./examples/warped_straight_lines.jpg "Warp Example"
[image5]: ./examples/color_fit_lines.jpg "Fit Visual"
[image6]: ./examples/example_output.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Camera Calibration


The code for this step is contained in the first code cell of the IPython notebook located in "camera_calibration.py".

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

<img src="./camera_cal/calibration3.jpg" alt="drawing" style="width:300px;"/>
<img src="./calibration3.jpg" alt="drawing" style="width:300px;"/>

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.

For the distortion correction I have followed the following steps
1. I used the opencv's undistort function to correct the image. 
2. For this function I took the camera matrix and distortion coefficients from the previous step 

<img src="./test_images/test1.jpg" alt="drawing" style="width:300px;"/>
<img src="./output_images/test1_undistorder.jpg" alt="drawing" style="width:300px;"/>

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

I used a combination of color, magnitude, direction and gradient thresholds to generate a binary image (thresholding steps at lines 48 through 58 in `pipelines.py`. The utility function definitions are in `color_transformations.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)

<img src="./output_images/test1_undistorder.jpg" alt="drawing" style="width:300px;"/>
<img src="./output_images/test1_combined.jpg" alt="drawing" style="width:300px;"/>

#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `warp_image()`, which appears in lines 6 through 29 in the file `perspective_transform.py`.  The `warp_image()` function takes as inputs an image (`img`), In the function I chose source and destination points by inspecting the image.  I chose the hardcode the source and destination points in the following manner:



This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 575, 475      |  200, 0        | 
| 725, 475      |  200, 700      |
| 275, 675      | 1000, 0      |
| 1050, 675     | 1000, 700       |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

<img src="./output_images/test1_combined.jpg" alt="drawing" style="width:300px;"/>
<img src="./output_images/test1_warped.jpg" alt="drawing" style="width:300px;"/>

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

For finding the lane pixels I created two functions `find_lane_pixels(binary_warped, nwindows=9, margin=100, minpix=50)`  and `searchRemainingPolly(image, left_fit, right_fit, nonzerox, nonzeroy , margin=50)` in file `find_lanes.py`

The first function is to identify the initial x and y fitting in the lane using historgram which is calculated using the warped image from previous step.

Once we find the initial point then we can use it base and fit a second degree polynomial to identify the next lane points.

<img src="./output_images/test1_warped.jpg" alt="drawing" style="width:300px;"/>
<img src="./output_images/test1_histogram.jpg" alt="drawing" style="width:300px;"/>




#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The curvature and position of vehicle is calculated in the` get_curvature() in find_lanes.py`
For the radius of curvature I used the left and right fit points from the previous step.with the following 

`left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])

right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines function draw_lanes in my code in `find_lanes.py` in the function . 
Once we get the left and right lane points which we calculated using warped image we can plot this lanes on top of the original image. Here is an example of my result on a test image:

<img src="./output_images/test1_final.jpg" alt="drawing" style="width:300px;"/>
---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_video/project_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Currently for the perspective transform I select the image by manually inspecting the test image. This will fail if the road lanes are narrow or has some color issues. Also one more issue is with the finding lanes. There might be a chance of stuck with going in wrong lane if there is more break in color in a lane 
