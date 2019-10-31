# usage $ python 2_selectObject.py --image input.jpg
# import the necessary packages
import argparse
import cv2
import numpy as np
from PIL import Image
import glob

# global variance
frame_number = 0
directory = ''
hsv = 0
lower = 0
upper = 0
count = 45

# for function 'selectObject'
cropping = False
getROI = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0


# use in function 'selectObject'
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


def selectObject():
    # grab references to the global variables
    global x_start, y_start, x_end, y_end, cropping, getROI, hsv, lower, upper

    # initialize the list of reference points and boolean indicating
    # whether cropping is being performed or not
    # x_start, y_start, x_end, y_end = 0, 0, 0, 0
    refPt = []

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-i", "--image", required=True, help="Path to the image")
    args = vars(ap.parse_args())

    # load the image, clone it, and setup the mouse callback function
    image = cv2.imread(args["image"])
    clone = image.copy()

    cv2.namedWindow("Select Object")
    cv2.setMouseCallback("Select Object", click_and_crop)

    # keep looping until the 'q' key is pressed
    while True:

        i = image.copy()

        if not cropping and not getROI:
            cv2.imshow("Select Object", image)

        elif cropping and not getROI:
            cv2.rectangle(i, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Select Object", i)

        elif not cropping and getROI:
            cv2.rectangle(image, (x_start, y_start), (x_end, y_end), (0, 255, 0), 2)
            cv2.imshow("Select Object", image)

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
        # selected domain
        # cv2.imshow("ROI", roi)

        hsvRoi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        # print('min H = {}, min S = {}, min V = {}; max H = {}, max S = {}, max V = {}'.format(hsvRoi[:,:,0.min(), hsvRoi[:,:,1].min(), hsvRoi[:,:,2].min(), hsvRoi[:,:,0].max(), hsvRoi[:,:,1].max(), hsvRoi[:,:,2].max()))

        lower = np.array([hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
        upper = np.array([hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])
        selectedObject(clone, 0)


def selectedObject(clone, i):
    if i != 0:
        image_to_thresh = cv2.imread(directory + 'frame/frame%d.jpg' % i)
    else:
        image_to_thresh = clone
    hsv = cv2.cvtColor(image_to_thresh, cv2.COLOR_BGR2HSV)
    kernel = np.ones((3, 3), np.uint8)
    # for red color we need to masks.
    mask = cv2.inRange(hsv, lower, upper)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # output
    # cv2.imshow("Mask", mask)
    cv2.imwrite(directory + "object/object%d.jpg" % i, mask)
    cv2.waitKey(0)
    # close all open windows
    # cv2.destroyAllWindows()
    # select color and press 'C'


def coloring(i):
    coloring_input = cv2.imread(directory + 'object/object%d.jpg' % i)
    raw = coloring_input.copy()

    coloring_input[np.where((coloring_input == [255, 255, 255]).all(axis=2))] = [0, 255, 255]

    # cv2.imshow('coloring_output', coloring_input)
    # cv2.imshow('coloring_input', raw)

    cv2.imwrite(directory + "coloring_output/coloring_output%d.jpg" % i, coloring_input)


def remove_background(i):
    img = Image.open(directory + "coloring_output/coloring_output%d.jpg" % i)
    img = img.convert("RGBA")
    datas = img.getdata()

    newData = []

    for item in datas:
        if item[0] < 100 and item[1] < 100 and item[2] < 100:
            newData.append((item[0], item[1], item[2], 0))
        else:
            newData.append(item)
    img.putdata(newData)

    # cannot write mode RGBA as JPEG
    img.save(directory + "remove_background/remove_background%d.png" % i)


def add_image(i):
    # read image
    original = cv2.imread(directory + 'frame/frame%d.jpg' % i)
    change = cv2.imread(directory + "remove_background/remove_background%d.png" % i)

    # store information about change image
    rows, cols, channels = change.shape

    roi = change[0:rows, 0:cols]
    dst = cv2.add(original, change)
    # cv2.imshow("dst", dst)
    # 합쳐진 이미지를 원본 이미지에 추가.
    change[0:rows, 0:cols] = dst

    # cv2.imshow('res', change)
    cv2.waitKey(0)
    cv2.imwrite(directory + "add_output/add_output%d.png" % i, dst)


if __name__ == '__main__':
    selectObject()
    for i in range(count):
        if i != 0:
            selectedObject(None, i)
        coloring(i)
        remove_background(i)
        add_image(i)
    cv2.destroyAllWindows()
