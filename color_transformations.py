import numpy as np
import cv2
import matplotlib.pyplot as plt
def abs_sobel(img, orient='x',sobel_kernel=3):
    gray= cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sobel = None
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    if orient =='y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    abs_sobel = np.absolute(sobel)
    return abs_sobel

def abs_sobel_threshold(img, sobel_value, thresh=(0, 255)):
    #Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255 * sobel_value / np.max(sobel_value))
    #Create a mask of 1's where the scaled gradient magnitude is > thresh_min and < thresh_max
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    
    #Return this mask as your binary_output image
    return binary_output

def get_magnitude_threshold(sobel_x, sobel_y, thresh=(0, 255)):
    # Calculate the magnitude
    abs_sobel = np.sqrt(sobel_x ** 2 + sobel_y ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))
    # Create a binary mask where mag thresholds are met
    binary_output = np.zeros_like(scaled_sobel)
    binary_output[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output


def get_dir_threshold(sobel_x, sobel_y, thresh=(0, np.pi / 2)):
    absgraddir = np.arctan2(np.absolute(sobel_y), np.absolute(sobel_x))

    # Create a binary mask where direction thresholds are met
    binary_output = np.zeros_like(absgraddir)
    binary_output[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1

    # Return this mask as your binary_output image
    return binary_output

def hls_select(img, thresh=(0, 255)):
    # 1) Convert to HLS color space
    hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)

    # 2) Apply a threshold to the S channel
    # 3) Return a binary image of threshold result
    s_channel = hls[:, :, 2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1

    return binary_output

def combined_color_transforms(sobel_x_binary, sobel_y_binary, mag_binary, dir_binary, hls_binary):
    combined_binary = np.zeros_like(dir_binary)
    combined_binary[
        ((sobel_x_binary == 1) & (sobel_y_binary == 1)) | ((mag_binary == 1) & (dir_binary == 1)) | hls_binary == 1] = 1
    return combined_binary