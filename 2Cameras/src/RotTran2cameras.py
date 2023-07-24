import cv2
import numpy as np

from triangulate import *
# Define the number of inner corners in the chessboard pattern
num_corners_x = 8
num_corners_y = 6
square_size = 1
# Set the termination criteria for corner subpixel refinement
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ..., (8,5,0)
objp = np.zeros((num_corners_x * num_corners_y, 3), np.float32)
objp[:, :2] = np.mgrid[0:num_corners_x, 0:num_corners_y].T.reshape(-1, 2)
objp *= square_size
# Arrays to store object points and image points from all images
objpoints = []  # 3D points in the world coordinate system
imgpoints1 = []  # 2D points in image plane of camera 1
imgpoints2 = []  # 2D points in image plane of camera 2

# Load the camera calibration matrices
K1 = camera_matrix1
K2 = camera_matrix2

# Capture images from camera 1 and camera 2
img1 = cv2.imread('Calibration_ext//camera1.jpg')
img2 = cv2.imread('Calibration_ext//camera2.jpg')

# Convert images to grayscale
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# Find chessboard corners in the images
ret1, corners1 = cv2.findChessboardCorners(gray1, (num_corners_x, num_corners_y), None)
ret2, corners2 = cv2.findChessboardCorners(gray2, (num_corners_x, num_corners_y), None)

if ret1 and ret2:
    # Refine the corner locations
    corners1 = cv2.cornerSubPix(gray1, corners1, (11, 11), (-1, -1), criteria)
    corners2 = cv2.cornerSubPix(gray2, corners2, (11, 11), (-1, -1), criteria)

    # Draw and display the corners
    cv2.drawChessboardCorners(img1, (num_corners_x, num_corners_y), corners1, ret1)
    cv2.drawChessboardCorners(img2, (num_corners_x, num_corners_y), corners2, ret2)

    # Store the object points and image points
    objpoints.append(objp)
    imgpoints1.append(corners1)
    imgpoints2.append(corners2)

    # Perform stereo calibration
    ret, _, _, _, _, R, t, _, _ = cv2.stereoCalibrate(objpoints, imgpoints1, imgpoints2, K1, dist_coeffs1, K2, dist_coeffs2, gray1.shape[::-1])

    if ret:
        print("Rotation matrix (R):")
        print(R)
        np.save('Rot.npy' , R)
        print("Translation vector (t):")
        print(t)
        np.save('Tr.npy' , t)
    else:
        print("Stereo calibration failed.")
else:
    print("Chessboard corners not found in one or both images.")

# Display the images with detected corners
cv2.imshow('Image 1', img1)
cv2.imshow('Image 2', img2)
cv2.waitKey(0)
cv2.destroyAllWindows()
