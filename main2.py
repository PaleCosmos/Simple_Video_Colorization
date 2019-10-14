import numpy as np
import cv2

directory = 'resources/sample/'


def tracking():
    try:
        print("cam on")
        cap = cv2.VideoCapture(directory + 'sample1_.mp4')
    except Exception as e:
        print(e.__str__())
        return

    while True:
        ret, frame = cap.read()

        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        lower_color = np.array([0, 0, 0])
        upper_color = np.array([100, 255, 255])

        mask_color = cv2.inRange(hsv, lower_color, upper_color)

        res = cv2.bitwise_and(frame, frame, mask=mask_color)

        cv2.imshow('coloring', res)

        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            return


tracking()
