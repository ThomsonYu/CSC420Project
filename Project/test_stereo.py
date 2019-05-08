import cv2


imgL = cv2.imread('stereo/hit7_left.jpg', 0)
imgR = cv2.imread('stereo/hit7_right.jpg', 0)
stereo = cv2.StereoBM_create(numDisparities=160, blockSize=15)

disparity = stereo.compute(imgL,imgR)

cv2.imwrite('stereo/hit7_disparity.jpg', disparity)

