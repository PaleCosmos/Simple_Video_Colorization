# usage $ python oneImage.py --image input.jpg
# import the necessary packages
import argparse
import cv2
import numpy as np
from PIL import Image
import glob

# global variance for function 'selectObject'
cropping = False
getROI = False
x_start, y_start, x_end, y_end = 0, 0, 0, 0


# Function to extract frames
def FrameCapture(path):
    # Path to video file 
    vidObj = cv2.VideoCapture(path)

    # Used as counter variable 
    count = 0

    # checks whether frames were extracted 
    success = 1

    while success:
        # vidObj object calls read 
        # function extract frames 
        success, image = vidObj.read()

        # Saves the frames with frame-count 
        cv2.imwrite("resources/frame/frame%d.jpg" % count, image)

        count += 1


# Function to make video
def MakeVideo():
    img_array = []
    n = 0
    for n in range(85):
        for filename in glob.glob('resources/frame/frame%d.jpg' % n):
            img = cv2.imread(filename)
            height, width, layers = img.shape
            size = (width, height)
            img_array.append(img)
        n += 1

    out = cv2.VideoWriter('resources/frame/link_frame.avi', cv2.VideoWriter_fourcc(*'DIVX'), 15, size)

    for i in range(len(img_array)):
        out.write(img_array[i])

    out.release()


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
    global x_start, y_start, x_end, y_end, cropping, getROI

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
        print('min H = {}, min S = {}, min V = {}; max H = {}, max S = {}, max V = {}'.format(hsvRoi[:, :, 0].min(),
                                                                                              hsvRoi[:, :, 1].min(),
                                                                                              hsvRoi[:, :, 2].min(),
                                                                                              hsvRoi[:, :, 0].max(),
                                                                                              hsvRoi[:, :, 1].max(),
                                                                                              hsvRoi[:, :, 2].max()))

        lower = np.array([hsvRoi[:, :, 0].min(), hsvRoi[:, :, 1].min(), hsvRoi[:, :, 2].min()])
        upper = np.array([hsvRoi[:, :, 0].max(), hsvRoi[:, :, 1].max(), hsvRoi[:, :, 2].max()])

        image_to_thresh = clone
        hsv = cv2.cvtColor(image_to_thresh, cv2.COLOR_BGR2HSV)

        kernel = np.ones((3, 3), np.uint8)
        # for red color we need to masks.
        mask = cv2.inRange(hsv, lower, upper)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # output
        # cv2.imshow("Mask", mask)
        cv2.imwrite("output.jpg", mask)
        cv2.waitKey(0)
    # close all open windows
    # cv2.destroyAllWindows()
    # select color and press 'C'


def coloring():
    coloring_input = cv2.imread('output.jpg')
    raw = coloring_input.copy()

    coloring_input[np.where((coloring_input == [255, 255, 255]).all(axis=2))] = [0, 255, 255]

    # cv2.imshow('coloring_output', coloring_input)
    # cv2.imshow('coloring_input', raw)

    cv2.imwrite("coloring_output.jpg", coloring_input)


def remove_background():
    img = Image.open("coloring_output.jpg")
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
    img.save("remove_background.png")


def add_image():
    # read image
    original = cv2.imread('input.jpg')
    change = cv2.imread('remove_background.png')

    # store information about change image
    rows, cols, channels = change.shape

    roi = change[0:rows, 0:cols]
    dst = cv2.add(original, change)

    # 합쳐진 이미지를 원본 이미지에 추가.
    change[0:rows, 0:cols] = dst

    # cv2.imshow('res', change)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite("add_output.png", dst)


if __name__ == '__main__':
    # Calling the function
    directory = 'resources/sample/'
    FrameCapture(directory + 'output.mp4')
    MakeVideo()
    '''
    selectObject()
    coloring()
    remove_background()
    add_image()
'''
