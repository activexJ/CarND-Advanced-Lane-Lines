import cv2
import numpy as np



def warp_image(binary_img):
    image = np.copy(binary_img)
    
    # Select point by viewing on image
    top_left = [575, 480]
    top_right = [725, 480]
    bottom_left = [280, 675]
    bottom_right = [1050, 675]

    src = np.float32([top_left, bottom_left, top_right, bottom_right])

    dst = np.float32([[200, 0], [200, 700], [1000, 0], [1000, 700]])

    # Grab the image shape
    img_size = (binary_img.shape[1], binary_img.shape[0])

    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(image, M, img_size)

    # Return the resulting image and matrix
    return warped, M