import cv2
import sys
import matplotlib.pyplot as plt
import numpy as np
from skimage.measure import regionprops, label
import skimage.exposure
from skimage import util
from scipy import ndimage
import time

# Starting time of initialization
start_time = time.time()

# Approximate color of lemons for segmentation
greenUpper = np.array([255, 250, 80],np.uint8)
greenLower = np.array([130, 60, 0],np.uint8)
 
# Argument for input image
file_name = sys.argv[1]

# Reading the image using cv packahge
src = cv2.imread(file_name, 1)
s = src

# if the image is having a low contrast then print "low exposure" and optimize it using equalizehist()
if (skimage.exposure.is_low_contrast(src,upper_percentile=100)) == True:
	print("low exposure")
	src = cv2.cvtColor(src, cv2.COLOR_BGR2YUV)
	src[:,:,0] = cv2.equalizeHist(src[:,:,0])
	src = cv2.cvtColor(src, cv2.COLOR_YUV2RGB)

# Do the color conversions	
src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
hsv = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)

# Segment the objects that lies between the color pallette
mask = cv2.inRange(src, greenLower, greenUpper)
src = mask

# Apply bilateral filter
src = cv2.bilateralFilter(src, 5, 175, 175)

# Finding Binary and OTSU threshold
ret2,th2 = cv2.threshold(src,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

# Detecting the edges
edge = cv2.Canny(th2, 10, 100)

# Finding the contours
image, contours, hierarchy = cv2.findContours(edge, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# Finding the distance transform between the contours
dist_transform = cv2.distanceTransform(th2,cv2.DIST_L2,5)
number = str(np.max(label(dist_transform > 0.18*dist_transform.max())))

# Result output
print("No of Lemons: "+number)
print("--- %s seconds ---" % (time.time() - start_time))
cv2.imshow(file_name, s)                                   # Display
cv2.waitKey()

# Another approch to count lemons
# contour_list = []
# for contour in contours:
#     approx = cv2.approxPolyDP(contour,0.01*cv2.arcLength(contour,True),True)
#     area = cv2.contourArea(contour)
#     if(len(contours)>100):
# 	    if (area > 30 ):
# 	        contour_list.append(contour)

# cv2.imshow('bg', dist_transform)                                  
# cv2.waitKey()