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
# logging.basicConfig(level=logging.DEBUG)


nx = 9
ny = 6
def findCorners():
    images = glob.glob('./camera_cal/calibration*.jpg')
    img_points = []
    object_points = []
    obj_point = np.zeros((ny*nx, 3), np.float32)
    obj_point[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    for imageName in images:    
        img = cv2.imread(imageName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
        if ret == True:
            img_points.append(corners)
            object_points.append(obj_point)
    return (object_points, img_points)

def undistort_image(image, camera_matrix, dist):
    return cv2.undistort(image,mtx,dist,None,mtx)

def findCalibrationParams(image, object_points, image_points):
    return cv2.calibrateCamera(object_points, image_points, image.shape[1::-1],None,None)
    
object_points, img_points = findCorners()
image = cv2.imread('./calibration1.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
ret, mtx,dist,rvecs,tvecs = findCalibrationParams(image, object_points, img_points)
udist = cv2.undistort(image,mtx,dist,None,mtx)



cv2.imshow("",udist)
cv2.waitKey(0)
#     else:
#         print("Error")