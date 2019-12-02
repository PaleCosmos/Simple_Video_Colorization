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
blank_image = None
s = False
h = None
w = None

hsv_boundaries_0 = [
    ([0, 70, 50], [10, 255, 255])
]
hsv_boundaries_1 = [
    ([170, 70, 50], [180, 255, 255])
]

def onMouse(event, x, y, flags, param):
    global col, width, row, height, frame, frame2, inputMode, blank_image, s, h, w
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
            # selection range is white, other is black
            blank_image = np.zeros((h, w, 3), np.uint8)
            blank_image[:] = (0, 0, 0)
            blank_image[row:row + height, col:col + width] = (255, 255, 255)
            s = True
    return


def camShift():
    # grab references to the global variables
    global frame, frame2, inputMode, trackWindow, roi_hist, roi, boundaries, blank_image, s, h, w

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
    lower0, upper0 = hsv_boundaries_0[0]
    lower0 = np.array(lower0, dtype="uint8")
    upper0 = np.array(upper0, dtype="uint8")
    lower1, upper1 = hsv_boundaries_1[0]
    lower1 = np.array(lower1, dtype="uint8")
    upper1 = np.array(upper1, dtype="uint8")
    
    
    while True:
        # read decode frame
        ret, frame = cap.read()
        if not ret:
            break

        # frame height, width
        h, w = frame.shape[:2]

        HSV_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        F_h, F_s, F_v = cv2.split(HSV_frame)
        
        # 변경 색상
        ChangeColor = np.zeros((h, w, 3), np.uint8)
        ChangeColor[:] = (0, 255, 255)
        #cv2.imshow("bgr", ChangeColor)
        # 변경 색상을 HSV로 변환 후, H/S/V의 값으로 split
        HSV_ChangeColor = cv2.cvtColor(ChangeColor, cv2.COLOR_BGR2HSV)
        cv2.imshow("change color(HSV)", HSV_ChangeColor)
        CC_h, CC_s, CC_v = cv2.split(HSV_ChangeColor)

        white_image = np.zeros((h, w, 3), np.uint8)
        white_image[:] = (255, 255, 255)

        # 선택된 색상 영역을 하얗게 만든다
        # HSVframe에서 lower0 ~ upper0 영역을 white로 변환
        mask_range_0 = cv2.inRange(HSV_frame, lower0, upper0)
        mask_white_0 = cv2.bitwise_or(HSV_frame, white_image, mask=mask_range_0)
        # HSVframe에서 lower1 ~ upper1 영역을 white로 변환
        mask_range_1 = cv2.inRange(HSV_frame, lower1, upper1)
        mask_white_1 = cv2.bitwise_or(HSV_frame, white_image, mask=mask_range_1)
        # 두 영역을 모두 포함
        mask_white = cv2.bitwise_or(mask_white_0, mask_white_1)
        #cv2.imshow('mask_white', mask_white)

        white_px = np.asarray([255, 255, 255])
        black_px = np.asarray([0, 0, 0])
        #img_array = np.array(img)
        # 선택된 색상 영역과 선택된 영역을 하얗게 만든다
        if s:
            # 선택된 영역(하얀색)은 하얗게, 나머지 배경은 검정색으로 변환
            mask_white = cv2.bitwise_and(blank_image, mask_white)
            '''
            for r in range(h):
                for c in range(w):
                    px = mask_white[r][c]
                    if all(px == white_px):
                        mask_white[r][c] = black_px
                    if all(px == black_px):
                        mask_white[r][c] = white_px
            cv2.imshow('mask_black', mask_white)
            frame = cv2.bitwise_or(frame, mask_white)
            '''
            #cv2.imshow('mask_white', mask_white)
            mask_black = cv2.bitwise_not(mask_white)
            #cv2.imshow('mask_black', mask_black)
            frame = cv2.bitwise_and(frame, mask_black)
            # 변환하고자 하는 색상의 H/S와 원래의 V(HSV_frame의 V)를 합친다
            merge_color = cv2.merge([CC_h, CC_s, F_v])
            #cv2.imshow('merge', merge_color)
            # 변환한 색상과 까만 배경
            merge_color = cv2.bitwise_and(merge_color, mask_white)
            
            # 변환한 사진을 BGR로 변환
            merge_color = cv2.cvtColor(merge_color, cv2.COLOR_HSV2BGR)
            #cv2.imshow('merge_color', merge_color)
            # 이를 frame과 합쳐, frame의 배경과 merge_color의 색상 변경 결과를 저장
            merge_result = cv2.bitwise_or(frame, merge_color)
            cv2.imshow('merge_result', merge_result)

        if trackWindow is not None:
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            dst = cv2.calcBackProject([hsv], [0], roi_hist, [0, 180], 1)
            ret, trackWindow = cv2.CamShift(dst, trackWindow, termination)
            pts = cv2.boxPoints(ret)
            pts = np.int0(pts)
            # roi is white, other is black
            blank_image[:] = (0, 0, 0)
            blank_image[np.min(pts[:, 1]):np.max(pts[:, 1]), np.min(pts[:, 0]):np.max(pts[:, 0])] = (255, 255, 255)
            
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


# main
if __name__ == '__main__':
    camShift()
