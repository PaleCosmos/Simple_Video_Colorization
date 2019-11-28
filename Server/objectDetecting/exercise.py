import numpy as np
import cv2
import math

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
    ([0, 0, 50], [80, 80, 255])
]

cvt2Colors = (255,0,0)


def calculate(a, b, c, x, y):
    return abs(a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)

def distance(a, b, c, d):
    return math.sqrt((a - c) ** 2 + (b - d) ** 2)

def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputMode, blank_image3, s, h, w
    global rectangle, roi_hist, trackWindow

    if inputMode:
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x, y

        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame, (col, row), (x, y), (255, 255, 255), 2)
                cv2.imshow('frame', frame)

        elif event == cv2.EVENT_LBUTTONUP:
            inputMode = False
            rectangle = False
            cv2.rectangle(frame, (col, row), (x, y), (255, 255, 255), 2)
            height, width = abs(row - y), abs(col - x)
            trackWindow = (col, row, width, height)
            roi = frame[row:row + height, col:col + width]
            roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roi_hist = cv2.calcHist([roi], [0], None, [180], [0, 180])
            cv2.normalize(roi_hist, roi_hist, 0, 255, cv2.NORM_MINMAX)
            blank_image3 = np.zeros((h, w, 3), np.uint8)
            blank_image3[:] = (0, 0, 0)
            blank_image3[row:row + height, col:col + width] = (255, 255, 255)
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

    # set window name 'frame' 
    cv2.namedWindow('frame')
    # check mouse event, and callback
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    # drive the meanShift back to 10 times or until the difference between C1_o and C1_r is 1pt.
    termination = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # color lower, upper
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    while True:
        # read decode frame
        ret, frame = cap.read()
        if not ret:
            break

        # frame height, width
        h, w = frame.shape[:2]

        mask = cv2.inRange(frame, lower, upper)  # 색부분
        mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        #if s:
            #mask2 = cv2.bitwise_and(blank_image3, mask2)

        
        resf = cv2.subtract(frame, mask2)
        #cv2.imshow('resf', resf)
        # set blue image
        blank_image = np.zeros((h, w, 3), np.uint8)
        blank_image[:] = cvt2Colors

        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2HSV)
        # set white image
        blank_image2 = np.zeros((h, w, 3), np.uint8)
        blank_image2[:] = (255, 255, 255)

        #V_Equals = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # print(V_Equals.shape)

        ppap = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        blank_image[:, :, 1] = ppap[:, :, 1]
        blank_image[:, :, 2] = ppap[:, :, 2]
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_HSV2BGR)
        # print(blank_image.shape)

        # set the color range
 
        # in mask == mask, frame or bloack_image2 (set white)
        trim1 = cv2.bitwise_or(frame, blank_image2, mask=mask)  # 몸통이하양
        if s:
            trim = cv2.bitwise_and(blank_image3, trim1)
            # in mask == mask, frame or bloack_image (set color(blue))
            trim2 = cv2.bitwise_and(trim, blank_image)  # 몸통이파랑 
            # frame and trim; background image(not object)
            trim3 = cv2.bitwise_and(frame, trim)  # 3ㅇㅔ서 trim
            # trim3 xor frame;
            trim4 = cv2.bitwise_xor(trim3, frame)

            #resf = cv2.subtract(frame, mask2)

            mask3 = cv2.bitwise_and(blank_image3, mask2)
            resf = cv2.subtract(frame, mask3)
            mask3 = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
            
            frame2 = cv2.bitwise_or(resf, blank_image, mask = mask3)
            frame2 = cv2.bitwise_or(resf, frame2)
        
            #cv2.imshow('frame2', resf)

        else:
            # trim = cv2.bitwise_and(blank_image3, trim1)
            # in mask == mask, frame or bloack_image (set color(blue))
            trim2 = cv2.bitwise_and(trim1, blank_image)  # 몸통이파랑
            # cv2.imshow("btrim2", trim2)
            # frame and trim; background image(not object)
            trim3 = cv2.bitwise_and(frame, trim1)
            # cv2.imshow("btrim3", trim3)
            # trim3 xor frame; 
            trim4 = cv2.bitwise_xor(trim3, frame)
            # cv2.imshow("btrim4", trim4)
            #frameNot = cv2.bitwise_and(frame, cv2.cvtColor(trim2, cv2.COLOR_HSV2BGR))
            frame2 = cv2.bitwise_or(resf, blank_image, mask = mask)
            frame2 = cv2.bitwise_or(resf, frame2)
        
        

        #frame2 = cv2.cvtColor(frame2, cv2.COLOR_HSV2BGR)

        # frame = cv2.bitwise_xor(frame, trim2)

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)

            blank_image3[:] = (0, 0, 0)
            blank_image3[np.min(pts[:, 1]):np.max(pts[:, 1]), np.min(pts[:, 0]):np.max(pts[:, 0])] = (255, 255, 255)

            #for dot in pts[:]:
            #    blank_image3[dot[1], dot[0]] = (255, 255, 255)

            # roi is white, other is black
            # print(pts[:])

            # print(pts[:, :])
            # dist1 = distance(pts[0][0], pts[0][1], pts[1][0], pts[1][1])
            # dist2 = distance(pts[0][0], pts[0][1], pts[2][0], pts[2][1])
            # dist3 = distance(pts[0][0], pts[0][1], pts[3][0], pts[3][1])

            # sumDist = dist1 + dist2 + dist3 - np.max([dist1, dist2, dist3])

            # #print(sumDist)

            # dots = []

            # maxSizeX = np.max(pts[:, 0]) - np.min(pts[:, 0])
            # maxSizeY = np.max(pts[:, 1]) - np.min(pts[:, 1])

            # for i, d1 in enumerate(pts):
            #     print('i is ', str(i), 'and pts is ', str(d1))
            #     for j, d2 in enumerate(pts):
            #         if i < j and not (abs(d1[0] - d2[0]) == maxSizeX or abs(d1[1] - d2[1]) == maxSizeY):
            #             dots.append((
            #                 d2[1] - d1[1],
            #                 d1[0] - d2[0],
            #                 (d1[1] * (d2[0] - d1[0]) - d1[0] * (d2[0] - d1[0]))
            #             ))

            # # dots = sorted(dots, key=lambda key: (-1 * key[0] / key[1]))
            # #print(dots)

            # for dot in blank_image3[np.min(pts[:, 1]):np.max(pts[:, 1]), np.min(pts[:, 0]):np.max(pts[:, 0])]:
            #     mLen = 0.0
            #     for v in dots:
            #         print(v)
            #         mLen += calculate(v[0], v[1], v[2], dot[0], dot[1])[0]
            #     if mLen <= sumDist:
            #         blank_image3[dot[0], dot[1]] = (255, 255, 255)

            #cv2.polylines(frame2, [pts], True, (0, 255, 0), 2)
            #cv2.imshow("blank", blank_image)
            
            # print(abc)
        if s:
            cv2.imshow('frame', frame2)
        else :
            cv2.imshow('frame', frame)
        
        
        # cv2.imshow('frameN', frameNot)
        # cv2.imshow('trim4', trim4)
        # cv2.imshow('trim', trim1)
        # cv2.imshow('b', trim4)
        # cv2.imshow('blank', blank_image)

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


# main
if __name__ == '__main__':
    camShift()
