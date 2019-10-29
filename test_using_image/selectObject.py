# usage $ python selectObject.py --image input.jpg
# import the necessary packages
import argparse
import cv2
import numpy as np
 
# initialize the list of reference points and boolean indicating
# whether cropping is being performed or not
x_start, y_start, x_end, y_end = 0, 0, 0, 0
cropping = False
getROI = False
refPt = []

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the image")
args = vars(ap.parse_args())
 
# load the image, clone it, and setup the mouse callback function
image = cv2.imread(args["image"])
clone = image.copy()


def click_and_crop(event, x, y, flags, param):
	# grab references to the global variables
	global x_start, y_start, x_end, y_end, cropping, getROI

	# if the left mouse button was clicked, record the starting
	# (x, y) coordinates and indicate that cropping is being
	# performed
	if event == cv2.EVENT_LBUTTONDOWN:
		x_start, y_start, x_end, y_end = x, y, x, y
		cropping = True

	elif event == cv2.EVENT_MOUSEMOVE:
		if cropping == True:
			x_end, y_end = x, y

	# check to see if the left mouse button was released
	elif event == cv2.EVENT_LBUTTONUP:
		# record the ending (x, y) coordinates and indicate that
		# the cropping operation is finished
		x_end, y_end = x, y
		cropping = False
		getROI = True

 
cv2.namedWindow("image")
cv2.setMouseCallback("image", click_and_crop)
 
# keep looping until the 'q' key is pressed
while True:

	i = image.copy()

	if not cropping and not getROI:
		cv2.imshow("image", image)

	elif cropping and not getROI:
		cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
		cv2.imshow("image", i)

	elif not cropping and getROI:
		cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
		cv2.imshow("image", image)

	key = cv2.waitKey(1) & 0xFF
 
	# if the 'r' key is pressed, reset the cropping region
	if key == ord("r"):
		image = clone.copy()
		getROI = False
 
	# if the 'c' key is pressed, break from the loop
	elif key == ord("c"):
		break


 
# if there are two reference points, then crop the region of interest
# from teh image and display it
refPt = [(x_start, y_start), (x_end, y_end)]
if len(refPt) == 2:
	roi = clone[refPt[0][1]:refPt[1][1], refPt[0][0]:refPt[1][0]]
	cv2.imshow("ROI", roi)

	hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
	print('min H = {}, min S = {}, min V = {}; max H = {}, max S = {}, max V = {}'.format(hsvRoi[:,:,0].min(), hsvRoi[:,:,1].min(), hsvRoi[:,:,2].min(), hsvRoi[:,:,0].max(), hsvRoi[:,:,1].max(), hsvRoi[:,:,2].max()))
 
	lower = np.array([hsvRoi[:,:,0].min(), hsvRoi[:,:,1].min(), hsvRoi[:,:,2].min()])
	upper = np.array([hsvRoi[:,:,0].max(), hsvRoi[:,:,1].max(), hsvRoi[:,:,2].max()])

	image_to_thresh = clone
	hsv = cv2.cvtColor(image_to_thresh, cv2.COLOR_BGR2HSV)

	kernel = np.ones((3,3),np.uint8)
	# for red color we need to masks.
	mask = cv2.inRange(hsv, lower, upper)
	mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
	mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

	cv2.imshow("Mask", mask)
	cv2.imwrite("output.jpg", mask)
	cv2.waitKey(0)
# close all open windows
cv2.destroyAllWindows()

# select color and press 'C'
