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
[image4]: ./writeup/hull_and_perspective_transform.png "Warp Example"
[image5]: ./writeup/warped_binary.png "Warped Binary Image"
[image6]: ./writeup/bounded_lane.png "Polynomial Fit example"
[image7]: ./examples/output_example.png "Output"
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

Applying the distortion removal to the pictures from the car was interesting. Notably, the areas on the outer edge of the view were shifted a great deal. This turned out to be important for a later step when I forgot to apply distoration removal as I calibrated a step and had to spend half a day redoing the work.

#### The image below shows the undistortion applied to a road image. 
![alt text][image2]
####2. Describe how (and identify where in your code) you used color transforms and gradients. Provide an example of a binary image result.

I applied magnitude and angle orientation primarily to identify dashed white lines in an image. Relevant function is on line 108 of colors.py.

To always identify yellow lines, especially on the white background of the bridge, it was important to use saturation. The function on line 29 of colors.py contains the threshold I used.

#### The image below shows the transform that I ended up applying in the project video. Notably, it deals quite well with yellow lanes. 
![alt text][image3]

####3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

When I started the project, I suspected that given the videos and the fact that the road curves gently, I could use a constant region to describe the perspective transform. However, during my experimentation, it appeared that the approximation didn't always hold up. In retrospect, I might say that this was the incorrect conclusion and I could have made it work. In any case, this conclusion led me to create an elaborate framework for detecting parallel lines and calculating their perspective transforms on the fly. The idea is as follows: because we are on a road, there will be many lines. But how does one pick out the relevant lines? The lines that we care about will be long and start in certain parts of the picture. They will have a restricted possible set of angles and will converge towards some vanishing point. With this in mind, I constructed two filters. One takes the lines from a Hough transform and picks out those lines with a certain slope and a certain starting point. This filter is defined on lines 78-170 in the file variable_perspective.py.

The second filter used the accumulator idea described in the hough transform. I specified a region which probably contained the vanishing point. Then I counted the lines as they travled through this region. Those cells with a high number of lines indicated lines which reached for a vanishing point. The function which accomplishes this is in line 227 of the file variable_perspective.py.

Finally, taking all the detected lines, I did a weighted average which favored long lines and lines close to the lower left and middle right (I used (150, 720) and (820, 720)). This function is on line 311 in variable_perspective.py.

This procedure caused the transformations to vary widely and the lane lines to move around a lot in the transformed plane. This was because as you passed dashed lines, their length changed rapidly, especially when a dashed line disappeared behind the car. This mean that the transformation changed a lot from frame to frame. Because the lines weren't parallel in the transformed plane at the starting point, the fitting algorithm was then vulnerable to blank spaces and often switched to lines it wasn't supposed to switch to.

#### The picture below is an example of the perspective transform applied to a lane. Note how the line isn't close to being perpendicular to the bottom of the frame. This was a persistent issue with my procedure. The procedure contained a great deal of noise that cuased the transform to jitter a great deal. This could have been partly fixed by playing around with smoothing on the hulls. 
![alt text][image4]

####4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Lane line pixels were identified with the convolution approach. The steps to this algorithm are as follows. First, identify a starting position of the line by bucketing all the activations in the lower forth of the picture. Then you slide a convolutional window across these bucekts to find the point with the largest number of on pixels. Record that point. Using this starting point, you then search 100 pixels in either direction around that point for the point of highest activation. Record that point. Search around this most recently found point in a window of 100 pixels in the same fashion and record. Repeat the last two steps until you reach the top under some pre-defined height step size.

Taking the points of highest activation from the convolution, I then fit a polynomial. The convolution and the fitting of the polynomial are accomplished on lines 28 and 140 with the functions find_window_centroids and bound_lanes.

#### The images below show first the transformed perspective of some lane lines and then the fitted bounding polynomial on the output to this. Note again how the lanes are not situated close to parallel. Nonetheless, the map down to the original space works well. 
![alt text][image5]
![alt text][image6]

####5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

The radius of curvature is found on lines 129 with the function radius_curvature in the file find_lanelines.py and the position of the vehicle with respect to center is found  on line 230 with the function distance_from_center_lane. 

The radius of curvature function probably applies the fact that locally, parabolas are like circles. Thus, you can approximate the distance to the center of a circle given the constants in a parabola by taking a variable circle and picking the radius which minimizes the error between the circle and the parobolic region it touches. 

####6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

The parobolic lanes are mapped back down into the original space in lines 159 and 160 of the file test_pipeline.py. 

![alt text][image7]

---

###Pipeline (video)

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result][video1] (The video should be viewable, but note that it is unlisted.)

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The main issue in my implementation is that it is probably more complex than necessary. On the bright side, it gives a relatively flexible framework that can be adapted to more extreme situations. The fact that it discovers perspective transformations ont he fly is useful for more difficult situations likea roads with extreme curves.

The other flaw in the implementation is that it is very sensitive to noise. I mentioned above the issue from the lane jiggling around a lot and even being unstable due to the fact that the dashed lines disappear every few frames. Similarly, if the lane lines fade into the background too much, the algorithm can pick out a fake lane lane-- e.g. the side of the road. 

The process for dealing with noise is itself quite fragile. The right sequence of images will easily result in an incorrect lane line. An example is found in the final frames of the video where the algorithm does not detect quickly enough that the lanes have become straight again and the ends of the lanes are off. 

The simplest way to fix the problem is to  just use a smaller region of interest and try to find lines in that area that reach for a vanishing point. Taking into account the radius of curvature in the previous frame, it is then possible to predict were the lines will converge to. This gives several simple and robust guards to change. The algorithm will be stable and much simpler than my implementation. 