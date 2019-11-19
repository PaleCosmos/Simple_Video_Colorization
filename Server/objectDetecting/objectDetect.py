import numpy as np
import cv2

col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputMode = False
rectangle = False
trackWindow = None
roi_hist = None
roi = None
s = False

boundaries = [
    ([0, 0, 50], [80, 80, 255])
]


def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputMode
    global rectangle, roi_hist, trackWindow

    if inputMode:
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
                cv2.imshow('frame', frame)

        elif event == cv2.EVENT_LBUTTONUP:
            inputMode = False
            rectangle = False
            cv2.rectangle(frame, (col, row), (x, y), (0, 255, 0), 2)
            height, width = abs(row - y), abs(col - x)
            trackWindow = (col, row, width, height)
            roi = frame[row:row + height, col:col + width]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            s = True
    return


def camShift():
    global frame, frame2, inputMode, trackWindow, roi_hist, roi, boundaries

    try:
        cap = cv2.VideoCapture('sampleVideo.mp4')
        cap.set(3, 480)
        cap.set(4, 320)
    except:
        print('Cam Failed')
        return

    ret, frame = cap.read()

    cv2.namedWindow('frame')
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        lower, upper = boundaries[0]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        h, w = frame.shape[:2]

        blank_image = np.zeros((h, w, 3), np.uint8)
        blank_image[:] = (0, 255, 0)

        blank_image2 = np.zeros((h, w, 3), np.uint8)
        blank_image2[:] = (255, 255, 255)
        blank_image3 = np.zeros((h, w, 3), np.uint8)
        blank_image3[:] = (0, 0, 0)

        mask = cv2.inRange(frame, lower, upper)  # 색부분

        trim = cv2.bitwise_or(frame, blank_image2, mask=mask)  # 몸통이하양

        trim2 = cv2.bitwise_and(trim, blank_image, mask=mask)  # 몸통이파랑

        trim2_ = cv2.cvtColor(trim2, cv2.COLOR_BGR2HSV)

        trim2_[:, :, 1] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 1]
        trim2_[:, :, 2] = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)[:, :, 2]

        trim2 = cv2.cvtColor(trim2_, cv2.COLOR_HSV2BGR)

        trim3 = cv2.bitwise_and(frame, trim)

        trim4 = cv2.bitwise_xor(trim3, frame)

        frameNot = cv2.bitwise_and(frame, trim2)

        frame_ = cv2.bitwise_or(trim4, trim2, mask=mask)

        frame = cv2.bitwise_or(frame_, frame)

        # frame = cv2.bitwise_xor(frame, trim2)

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)

            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow('frame', frame)
        cv2.imshow('trim2', trim2_)
        cv2.imshow('trim4', trim4)

        k = cv2.waitKey(100) & 0xFF
        if k == 27:
            break

        if k == ord('i'):
            print('추적한 영역을 지정하고 아무키나 누르세요')
            inputMode = True
            frame2 = frame.copy()

            while inputMode:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


camShift()
