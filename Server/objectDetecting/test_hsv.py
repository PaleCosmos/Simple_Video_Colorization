import numpy as np
import cv2

# global variance
col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputMode = False
rectangle = False
trackWindow = None
roi_hist = None
roi = None
blank_image3 = None
s = False
h = None
w = None

boundaries = [
    ([0, 0, 50], [100, 100, 255])
]


def onMouse(event, x, y, flags, param):
    # grab references to the global variables
    global col, width, row, height, frame, frame2, inputMode, blank_image3, s, h, w
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
            # set blank_image
            blank_image3 = np.zeros((h, w, 3), np.uint8)
            blank_image3[:] = (0, 0, 0)
            blank_image3[row:row + height, col:col + width] = (255, 255, 255)
            # switch on
            s = True
    return


def camShift():
    # grab references to the global variables
    global frame, frame2, inputMode, trackWindow, roi_hist, roi, boundaries, blank_image3, s, h, w

    try:
        # read video
        cap = cv2.VideoCapture('sampleVideo.mp4')
        # set the size of screen
        cap.set(3, 480)
        cap.set(4, 320)
    except:
        print('Cam Failed')
        return

    # read decode frame
    ret, frame = cap.read()
    cv2.namedWindow('frame')
    # check mouse event, and callback
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    # drive the meanShift back to 10 times or until the difference between C1_o and C1_r is 1pt.
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    while True:
        # read decode frame
        ret, frame = cap.read()
        if not ret:
            break

        # color lower, upper
        lower, upper = boundaries[0]
        lower = np.array(lower, dtype="uint8")
        upper = np.array(upper, dtype="uint8")

        # frame height, width
        h, w = frame.shape[:2]

        # set blue image
        blank_image = np.zeros((h, w, 3), np.uint8)
        blank_image[:] = (255, 0, 0)
        # set white image
        blank_image2 = np.zeros((h, w, 3), np.uint8)
        blank_image2[:] = (255, 255, 255)

        V_Equals = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # print(V_Equals.shape)
        blank_image[:, :, 1] = V_Equals[:, :, 1]
        blank_image[:, :, 2] = V_Equals[:, :, 2]
        # print(blank_image.shape)

        # set the color range
        mask = cv2.inRange(frame, lower, upper)  # 색부분
        # in mask == mask, frame or bloack_image2 (set white)
        trim = cv2.bitwise_or(frame, blank_image2, mask=mask)  # 몸통이하양
        if s:
            trim = cv2.bitwise_and(blank_image3, trim)
        # in mask == mask, frame or bloack_image (set color(blue))
        trim2 = cv2.bitwise_and(trim, blank_image)  # 몸통이파랑
        # frame and trim; background image(not object)
        trim3 = cv2.bitwise_and(frame, trim)
        # trim3 xor frame;
        trim4 = cv2.bitwise_xor(trim3, frame)

        frameNot = cv2.bitwise_and(frame, trim2)

        frame = cv2.bitwise_or(trim4, trim2)

        # frame = cv2.bitwise_xor(frame, trim2)

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)

            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)

            # roi is white, other is black
            blank_image3[:] = (0, 0, 0)
            blank_image3[np.min(pts[:, 1]):np.max(pts[:, 1]), np.min(pts[:, 0]):np.max(pts[:, 0])] = (255, 255, 255)

            cv2.polylines(frame, [pts], True, (0, 255, 0), 2)

        cv2.imshow('frame', frame)

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
