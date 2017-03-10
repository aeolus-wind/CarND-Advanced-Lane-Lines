##Writeup Template
###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

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

[image1]: ./writeup_images/undistorted_output.png "Undistorted"
[image2]: ./writeup_images/undistorted_road.jpg "Road Transformed"
[image3]: ./writeup_images/binary_combo_example.jpg "Binary Example"
[image4]: ./writeup_images/hull_and_perspective_transform.png "Warp Example"
[image5]: ./writeup_images/warped_binary.png "Warped Binary Image"
[image6]: ./writeup_images/bounded_lane.png "Polynomial Fit example"
[image7]: ./writeup_images/output_example.png "Output"
[video1]: https://youtu.be/fUakW3wDKxA "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

###Camera Calibration

####1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The distortion was removed by first calibrating a transformation on a chess board. To elaborate, pictures of the chessboard were taken from different angles. Because we know how the different squares in a chessboard relate to each other in size, we can use the information in the picture of the chessboard in different situations to determine how much distortion comes from a particular lens.

The code that accomplishes this is on lines 31 and 54. In line 31, the points on the chess board are discovered. These are called image points because they are where the chess board points are in the image. Then we define arbitrary points which describe the true grid structure of a chess board. In this case, it's a 2-d cartesian product of the integers of appropriate range with a fixed 0 indicating that our hypothetical 2-d chess board exists in 3 dimensions. After this is defined, the function called in line 54 does some optimization to find the transformation to remove distortion.

It is notable, that in this process, something like 3 images failed with the parameters that I called it with. This was due to the fact that some points were cut off in the given chess-board images so that the given parameters didn't work. Despite this, the parameters I chose covered 85% of the pictures and gave a satisfactory result.


#### The image belows shows an undistorted chess board. Incidentally, it was one of the 3 that failed to have the right dimensions in the first step of the calibration process. Nonetheless the undistort function works fine.
![alt text][image1]

###Pipeline (single images)

####1. Provide an example of a distortion-corrected image.

Applying the distortion removal to the pictures from the car was interesting. Notably, the areas on the outer edge of the view were shifted a great deal. 

#### The image below shows the undistortion applied to a road image. 
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms and gradients. Provide an example of a binary image result.

I applied magnitude and angle orientation primarily to identify dashed white lines in an image. Relevant function is on line 108 of colors.py.

To always identify yellow lines, especially on the white background of the bridge, it was important to use saturation. The function on line 29 of colors.py contains the threshold I used.

#### The image below shows the transform that I ended up applying in the project video. Notably, it deals quite well with yellow lanes. 
![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The perspective transform was applied on lines 77-83. I assumed a constant region would be consistently rectangular in the birds'-eye view . 


| Source        | Destination   | 
|:-------------:|:-------------:| 
| 390, 600      | 455, 650      | 
| 940, 590      | 870, 670      |
| 560, 480      | 460, 350      |
| 745, 470      | 905, 360      |

I did a fair bit of experimentation over how narrow I should make the destination region. Everything under 300 pixels caused extreme distortion in the drawn lane lines


#### The picture below is an example of the perspective transform applied to a lane. Note how the line isn't close to being perpendicular to the bottom of the frame. This was a persistent issue with my procedure. The procedure contained a great deal of noise that cuased the transform to jitter a great deal. This could have been partly fixed by playing around with smoothing on the hulls. 
![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels were identified with the convolution approach. The steps to this algorithm are as follows. First, identify a starting position of the line by bucketing all the activations in the lower forth of the picture. Then you slide a convolutional window across these bucekts to find the point with the largest number of on pixels. Record that point. Using this starting point, you then search 100 pixels in either direction around that point for the point of highest activation. Record that point. Search around this most recently found point in a window of 100 pixels in the same fashion and record. Repeat the last two steps until you reach the top under some pre-defined height step size.

Taking the points of highest activation from the convolution, I then fit a polynomial. The convolution and the fitting of the polynomial are accomplished on lines 29 and 102 with the functions find_window_centroids and bound_lanes.

#### The images below show first the transformed perspective of some lane lines and then the fitted bounding polynomial on the output to this. Note again how the lanes are not situated close to parallel. Nonetheless, the map down to the original space works well. 
![alt text][image5]
![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is found on lines 129 with the function radius_curvature in the file find_lanelines.py and the position of the vehicle with respect to center is found  on line 167 with the function distance_from_center_lane. 

The radius of curvature function probably applies the fact that locally, parabolas are like circles. Thus, you can approximate the distance to the center of a circle given the constants in a parabola by taking a variable circle and picking the radius which minimizes the error between the circle and the parobolic region it touches. 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The parobolic lanes are mapped back down into the original space in lines 118 of the file test_pipeline.py. 

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1] (The video should be viewable, but note that it is unlisted.) It is also in the github repo. 

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

There are points in the video where my filtering is not perfect or is a little slow. This is probably because the restrictions on accepting a new image are a bit too stringent or it applies the simple physics-based smoothing too liberally. 

If there are multiple frames in a row with changes in lighting, the process will also fail as the filters do not remove shadow. 