import numpy as np
import cv2

directory = 'resources/sample/'

cap = cv2.VideoCapture(directory + 'sample.mp4')

output_size = (187, 333)

fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter(directory + 'output.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), output_size)

if not cap.isOpened():
    exit()

tracker = cv2.TrackerCSRT_create()

ret, img = cap.read()

cv2.namedWindow('Select Window')
cv2.imshow('Select Window', img)

# setting ROI
rect = cv2.selectROI('Select Window', img, fromCenter=False, showCrosshair=True)
cv2.destroyWindow('Select Window')

# initialize tracker
tracker.init(img, rect)

while True:
    ret, img = cap.read()

    if not ret:
        exit()

    success, box = tracker.update(img)

    left, top, w, h = [int(v) for v in box]

    center_x = left + w / 2
    center_y = top + h / 2

    result_top = int(center_y - output_size[1] / 2)
    result_bottom = int(center_y + output_size[1] / 2)
    result_left = int(center_x - output_size[0] / 2)
    result_right = int(center_x + output_size[0] / 2)

    result_img = img[result_top:result_bottom, result_left: result_right].copy()

    out.write(result_img)

    cv2.rectangle(img, pt1=(left, top), pt2=(left + w, top + h), color=(255, 255, 255),
                  thickness=3)

    # 객체를 추적한 frame show
    cv2.imshow('result_img', result_img)
    # 전체적인 frame show
    cv2.imshow('img', img)
    if cv2.waitKey(1) == ord('q'):
        break
