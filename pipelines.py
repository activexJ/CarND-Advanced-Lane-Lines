#stages
'''
    1. Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
    2. Apply a distortion correction to raw images.
    3. Use color transforms, gradients, etc., to create a thresholded binary image.
    4. Apply a perspective transform to rectify binary image ("birds-eye view").
    5. Detect lane pixels and fit to find the lane boundary.
    6. Determine the curvature of the lane and vehicle position with respect to center.
    7. Warp the detected lane boundaries back onto the original image.
    8. Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
'''

import numpy as np
import cv2
import logging
import glob
import matplotlib.pyplot as plt
# logging.basicConfig(level=logging.DEBUG)

import camera_calibration as cc
import color_transformations as ct
import perspective_transform as pt
import find_lanes as fl
import os

# Camera calibration
calibration_images_path = './camera_cal/calibration*.jpg'
nx,ny = 9, 6
object_points, img_points, img_shape = cc.findCorners(nx,ny, calibration_images_path)
ret, mtx,dist,rvecs,tvecs = cc.findCalibrationParams(img_shape, object_points, img_points)

left_fit = []
right_fit = []
isNew = True

calibration3Image = cv2.imread("./camera_cal/calibration3.jpg")
cv2.imwrite('calibration3.jpg',cc.undistort_image(calibration3Image, mtx, dist))

def findLane(image, isNew,image_name = None, writeToFile = False):

    #1. Removing distortion from the image
    undistorted_img = cc.undistort_image(image, mtx, dist)
    
    #2.  Calculating the sobel of the image for other transformations
    sobelx = ct.abs_sobel(undistorted_img)
    sobely = ct.abs_sobel(undistorted_img, orient='y')
    #3. Calculating the sobel thresholded image with both x & y
    sobel_x_binary = ct.abs_sobel_threshold(undistorted_img, sobelx, thresh=(20, 100))
    sobel_y_binary = ct.abs_sobel_threshold(undistorted_img, sobely, thresh=(20, 100))
    #4. Calculating the magnitude threshold, direction and making  image
    mag_threshold_binary = ct.get_magnitude_threshold(sobelx, sobely, thresh=(20, 100))
    dir_threshold_binary = ct.get_dir_threshold(sobelx, sobely, thresh=(0.7, 1.4))

    #5. Applying threshold in S channel. Need to check with other channels
    hls_threshold_binary = ct.hls_select(undistorted_img, thresh=(150, 255))

    #6. Combing the different transformed images
    combined_binary = ct.combined_color_transforms(sobel_x_binary, sobel_y_binary, mag_threshold_binary, dir_threshold_binary, hls_threshold_binary)
    
    #7. Applied perspective transform to the image
    warped_img, M = pt.warp_image(combined_binary)
    # test = fl.fit_polynomial(warped_img)
    # plt.imshow(test)
    # plt.show()
    '''
    Identinfing the lane points in the image. If it is a new image then use histogram to find the left fit and right fit
    Else use the previously identified pointts to calculate the next points
    '''
    if isNew:
        left_fit, right_fit, left_lane_inds, right_lane_inds, nonzerox, nonzeroy, out_img = fl.find_lane_pixels(warped_img)
        isNew = False
    else:
        left_fit, right_fit = fl.searchRemainingPolly(image, left_fit, right_fit, nonzerox, nonzeroy)

   
    # calculating the curvatures and center of the car using the left and right fit
    left_curvature, right_curvature, center = fl.get_curvature(warped_img, left_fit, right_fit)
    result = fl.draw_lanes(image, warped_img, left_fit, right_fit, M, left_curvature, right_curvature, center)
    
    if writeToFile and image_name is not None:
        writeToFiles(image, undistorted_img, combined_binary, warped_img, result, image_name)
    return result
    # plt.imshow(cv2.cvtColor(result,  cv2.COLOR_BGR2RGB))

    # plt.show()
def writeToFiles(org_image, undistorted_img, combined_binary, warped_img, final_img,image_name):
    result_path = 'output_images'
    filename = os.path.splitext(image_name)
    cv2.imwrite(os.path.join(os.getcwd(),result_path,filename[0]+'_org'+filename[1]),org_image) 
    cv2.imwrite(os.path.join(os.getcwd(),result_path,filename[0]+'_undistorder'+filename[1]),undistorted_img) 
    # plt.imshow(warped_img)
    # plt.show()
    cv2.imwrite(os.path.join(os.getcwd(),result_path,filename[0]+'_combined'+filename[1]),combined_binary * 255) 
    plt.plot(fl.createHistograms(warped_img)) 
    
    plt.savefig(os.path.join(os.getcwd(),result_path,filename[0]+'_histogram'+filename[1]))
    cv2.imwrite(os.path.join(os.getcwd(),result_path,filename[0]+'_warped'+filename[1]),warped_img * 255) 
    cv2.imwrite(os.path.join(os.getcwd(),result_path,filename[0]+'_final'+filename[1]),final_img) 

# for video processing
def process_image(image):
    return findLane(image, isNew)


# for test image processing
def process_test_images():
    images_path = 'test_images'
    result_path = 'output_images'
    for image_path in os.listdir(images_path):
        findLane(cv2.imread(os.path.join(os.getcwd(),images_path,image_path)), True, image_path, True)
        

    
if __name__ == '__main__':
    # image = cv2.imread('./test_images/test6.jpg')
    # result = findLane(image, isNew)
    # plt.imshow(result)
    # plt.show()
    #generate output images
    # process_test_images()

    from moviepy.editor import VideoFileClip
    
    white_output = 'output_video/project_video1.mp4'
    clip1 = VideoFileClip("project_video.mp4")

    white_clip = clip1.fl_image(process_image)
    white_clip.write_videofile(white_output, audio=False)
