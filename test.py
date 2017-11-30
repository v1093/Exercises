import numpy as np
import cv2
import yaml
import glob
import scipy.io as io
import matplotlib.pyplot as plt

# termination criteria
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
print(criteria)
# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((5*8,3), np.float32)
objp[:,:2] = np.mgrid[0:8,0:5].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.


found = 0
images = glob.glob('data/*.jpg')
for fname in images:  # Here, 10 can be changed to whatever number you like to choose
    img = cv2.imread(fname) # Capture frame-by-frame
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chess board corners
    ret, corners = cv2.findChessboardCorners(gray, (8,5), None)

    # If found, add object points, image points (after refining them)
    if ret == True:

        objpoints.append(objp)  # Certainly, every loop objp is the same, in 3D.

        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        #print(corners2)

        # Draw and display the corners
        img = cv2.drawChessboardCorners(img, (8,5), corners2, ret)
        found += 1

    cv2.imshow('img', img)
    cv2.waitKey()


print(found)
# When everything done, release the capture

cv2.destroyAllWindows()
#
#objpoints = np.array(objpoints ,dtype=np.float32)
#imgpoints = np.array(imgpoints,dtype=np.float32)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
#print(dist)

img = cv2.imread('00000.jpg')

cv2.imshow('img', img)
cv2.waitKey()
h,  w = img.shape[:2]
newcameramtx, roi=cv2.getOptimalNewCameraMatrix(mtx,dist,(w,h),1,(w,h))

# undistort
mapx,mapy = cv2.initUndistortRectifyMap(mtx,dist,None,newcameramtx,(w,h),5)
dst = cv2.remap(img,mapx,mapy,cv2.INTER_LINEAR)
print(newcameramtx)
# crop the image
x,y,w,h = roi
dst = dst[y:y+h, x:x+w]
height, width = img.shape[:2]
out1 = cv2.resize(dst,  (width, height))
cv2.imwrite('calibresult.jpg',out1)


img1 = cv2.imread('calibresult.jpg')
gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
ret, corners = cv2.findChessboardCorners(gray1, (8,5), None)
#print(corners)
corners2 = cv2.cornerSubPix(gray1, corners, (11, 11), (-1, -1), criteria)
img2 = cv2.imread("00000.jpg")
img2 = cv2.drawChessboardCorners(img2, (8,5), corners2, ret)
#print(corners2)
cv2.imwrite("mapped.jpg", img2)
cv2.imshow('img', img2)
cv2.waitKey()

#
#
# # It's very important to transform the matrix to list.
# data = {'camera_matrix': np.asarray(mtx).tolist(), 'dist_coeff': np.asarray(dist).tolist()}
#
# with open("calibration.yaml", "w") as f:
#     yaml.dump(data, f)
#
#
# # You can use the following 4 lines of code to load the data in file "calibration.yaml"
# with open('calibration.yaml') as f:
#     loadeddict = yaml.load(f)
# mtxloaded = loadeddict.get('camera_matrix')
# distloaded = loadeddict.get('dist_coeff')
# print(mtxloaded)
# print("asdf")
# print(distloaded)
