import numpy as np
import cv2
import glob

def findCorners(nx, ny, imagesPath):
    images = glob.glob(imagesPath)
    img_points = []
    object_points = []
    obj_point = np.zeros((ny*nx, 3), np.float32)
    obj_point[:,:2] = np.mgrid[0:9,0:6].T.reshape(-1,2)
    img_shape = None
    for imageName in images:    
        img = cv2.imread(imageName)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_shape = img.shape[1::-1]
        ret, corners = cv2.findChessboardCorners(gray,(nx,ny),None)
        if ret == True:
            img_points.append(corners)
            object_points.append(obj_point)
    return (object_points, img_points, img_shape)

def undistort_image(image, camera_matrix, dist):
    return cv2.undistort(image,camera_matrix,dist,None,camera_matrix)


# returns ret, mtx,dist,rvecs,tvecs 
def findCalibrationParams(img_shape, object_points, image_points):
    return cv2.calibrateCamera(object_points, image_points, img_shape,None,None)