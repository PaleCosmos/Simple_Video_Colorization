import numpy as np
import cv2
import math
import copy

col, width, row, height = -1, -1, -1, -1
frame = None
frame2 = None
inputMode = False
inputMode2 = False
rectangle = False
trackWindow = None
roi_hist = None
upper = None
lower = None
roi = None
blank_image3 = None
s = False
h = None
w = None
clFlag = False
poip = 0
colorSum = [0,0,0]
fcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None


colorDifferenceValue = 255
colorDifferenceValues = [colorDifferenceValue,
                         colorDifferenceValue, colorDifferenceValue]

boundaries = [
    ([0, 0, 0], [255, 255, 255])
]

cvt2Colors = (255, 0, 255)

def calculate(a, b, c, x, y):
    return abs(a * x + b * y + c) / math.sqrt(a ** 2 + b ** 2)


def distance(a, b, c, d):
    return math.sqrt((a - c) ** 2 + (b - d) ** 2)


def onMouse(event, x, y, flags, param):
    global clFlag, col, width, row, height, frame, frame2, inputMode, inputMode2, blank_image3, s, h, w
    global rectangle, roi_hist, trackWindow, colorDifferenceValue, upper, lower
    global poip, colorSum
 
    if event == cv2.EVENT_LBUTTONDOWN:
        print(trackWindow is not None)
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
            print("색상 영역에서 드래그해주세여.")
            inputMode2 = True
            inputMode = False

    elif inputMode2:

        if not clFlag and event == cv2.EVENT_LBUTTONDOWN:
            clFlag = True
            print('downevent')
        if clFlag and event == cv2.EVENT_MOUSEMOVE:
            poip = poip + 1
            ku = frame[y,x].copy()
            colorSum = [colorSum[0] + ku[0],colorSum[1] + ku[1],colorSum[2] + ku[2]]

        if poip != 0 and event == cv2.EVENT_LBUTTONUP:
            print('upevent')
            clFlag = False

            mgr = [colorSum[0]/poip,colorSum[1]/poip,colorSum[2]/poip]

            print('mgr: ' + str(mgr))

            sortedValue = sorted(
                [0, 1, 2], key=lambda x: mgr[x], reverse=False)

            colorContectValue = mgr[0] + mgr[1] + mgr[2]

            print(sortedValue)
            los = []
            ups = []

            for k in sortedValue:
                colorDifferenceValues[k] = mgr[k] *colorDifferenceValue / colorContectValue

            for i in range(0, 3):
                los.append(mgr[i] - colorDifferenceValues[i]
                           if mgr[i] > colorDifferenceValues[i] else 0)
                ups.append(mgr[i] + colorDifferenceValues[i]
                           if mgr[i] < 255-colorDifferenceValues[i] else 255)

            boundaries[0] = (los, ups)
            lower = np.array(los, dtype="uint8")
            upper = np.array(ups, dtype="uint8")

            s = True
            inputMode2 = False
    return


def camShift():
    # grab references to the global variables
    global frame, frame2, inputMode, inputMode, trackWindow, roi_hist, roi, boundaries, blank_image3, s, h, w, upper, lower
    global fcc, out

    try:
        # read video
        cap = cv2.VideoCapture('sampleVideo.mp4')
        # set the size of screen
        cap.set(3, 480)
        cap.set(4, 320)
        wx = int(cap.get(3))
        hx = int(cap.get(4))
        out = cv2.VideoWriter('output.mp4', fcc, 20, (wx, hx))

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
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        mask = cv2.inRange(frame, lower, upper)  # 색부분
        mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        resf = cv2.subtract(frame, mask2)

        blank_image = np.zeros((h, w, 3), np.uint8)
        blank_image[:] = cvt2Colors

        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2HSV)
        blank_image2 = np.zeros((h, w, 3), np.uint8)
        blank_image2[:] = (255, 255, 255)

        ppap = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        blank_image[:, :, 2] = ppap[:, :, 2]
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_HSV2BGR)
  
        trim1 = cv2.bitwise_or(frame, blank_image2, mask=mask)  # 몸통이하양
        if s:
            trim = cv2.bitwise_and(blank_image3, trim1)
            mask3 = cv2.bitwise_and(blank_image3, mask2)
            resf = cv2.subtract(frame, mask3)
            mask3 = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.bitwise_or(resf, blank_image, mask=mask3)
            frame2 = cv2.bitwise_or(resf, frame2)

        else:
            trim3 = cv2.bitwise_and(frame, trim1)
            frame2 = cv2.bitwise_or(resf, blank_image, mask=mask)
            frame2 = cv2.bitwise_or(resf, frame2)
            
        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            copyPts = pts.copy()
            #print(ret)

            for (i, asdf) in enumerate(pts):
                if asdf[0] >=w:
                    copyPts[i][0] =w
                elif asdf[0] <=0:
                    copyPts[i][0] = 0
                if asdf[1] >=h:
                    copyPts[i][1] = h
                elif asdf[1] <=0:
                    copyPts[i][1] = 0

            blank_image3[:] = (0, 0, 0)
            blank_image3[np.min(copyPts[:, 1]):np.max(copyPts[:, 1]), np.min(
                copyPts[:, 0]):np.max(copyPts[:, 0])] = (255, 255, 255)
            writeFrame = frame2.copy()

            cv2.polylines(writeFrame, [copyPts], True, (0, 255, 0), 2)

        if s:
            cv2.imshow('frame', writeFrame)
            out.write(frame2)
        else:
            cv2.imshow('frame', frame)

        k = cv2.waitKey(100) & 0xFF

        if k == 27:
            break

        if k == ord('i'):
            print('추적한 영역을 지정하고 아무키나 누르세요')
            inputMode = True
            frame2 = copy.copy(frame)

            while inputMode or inputMode2:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)
        if k == ord('q'):
            print('종료되었습니다.')
            out.release()
            break

    cap.release()
    cv2.destroyAllWindows()

# main
if __name__ == '__main__':
    camShift()
