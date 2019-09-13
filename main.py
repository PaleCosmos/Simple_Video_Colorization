import cv2
import numpy as np

# 동영상의 디렉토리
directory = 'resources/sample/'

# 동영상 객체
capture = cv2.VideoCapture(directory + 'sample.mp4')

# 20000ms 이후를 보여줌
# capture.set(cv2.CAP_PROP_POS_MSEC, 20000)
# success, image = capture.read()

ret, frame1 = capture.read()

prvs = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)

hsv = np.zeros_like(frame1)

hsv[..., 1] = 255

while True:
    ret, frame2 = capture.read()
    next = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prvs, next, None, 0.5, 3, 15, 3, 5, 1.2, 0)

    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
    rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    cv2.imshow('frame2', rgb)
    k = cv2.waitKey(30) & 0xff
    if k == 27:
        break
    elif k == ord('s'):
        cv2.imwrite('opticalfb.png', frame2)
        cv2.imwrite('opticalhsv.png', rgb)
    prvs = next

capture.release()

# 스크린샷 저장
# if success:
#     cv2.imwrite(directory + 'sample.jpg', image)
#     cv2.imshow("20sec", image)
#     cv2.waitKey()

# while capture.isOpened():
#
#     ret, frame = capture.read()
#
#     if ret:
#         cv2.imshow("SVC", frame)
#         if cv2.waitKey(33) > 0:
#             break
#     else:
#         break


# capture.release()
cv2.destroyAllWindows()
