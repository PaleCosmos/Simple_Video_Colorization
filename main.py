import numpy as np
import cv2
import math
import copy
from data.trackers import tracker_create
from tkinter import *
from tkinter.colorchooser import *
import sys


# 나가기 버튼
def ExitButton():
    global main
    main.quit()

# 색 가져옴
def getColor():
    global cvt2Colors
    temp = askcolor()[0]
    cvt2Colors = np.array([temp[2], temp[1], temp[0]])
    
# closing 이벤트
def on_closing():
    global inputMode2, main, s, myTrackerType, variable
    # 마우스 이벤트 종료, trackwindow가 None이 아니므로 추적 및 Coloring 시작
    myTrackerType = variable.get()
    s = True
    inputMode2=False
    main.destroy()

# 데이터들
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
colorSum = [0, 0, 0]
fcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None
tracker_initation = True
main = None
variable = None

# opencv에서 제공해주는 알고리즘이 있었음
tracker_types = ['BOOSTING', 'MIL','KCF', 'TLD', 'MEDIANFLOW', 'CSRT', 'MOSSE']
myTrackerType = 'BOOSTING'

# 이거에 따라 정확도 달라지게 알고리즘 짰음
colorDifferenceValue = 255
colorDifferenceValues = [colorDifferenceValue,
                         colorDifferenceValue, colorDifferenceValue]

# 색 경계 초기화
boundaries = [
    ([0, 0, 0], [255, 255, 255])
]

# 바꾸고자 하는 색상 나중에 input으로 받아옵시다
cvt2Colors = (255, 0, 0)

# 박스를 지정하고 바꾸고자 하는 색상을 문질문질하면
# 색상의 평균값을 구해서 알고리즘에 대입함
def onMouse(event, x, y, flags, param):
    global clFlag, col, width, row, height, frame, frame2, inputMode, inputMode2, blank_image3, s, h, w
    global rectangle, roi_hist, trackWindow, colorDifferenceValue, upper, lower
    global poip, colorSum

    # 출력용
    if event == cv2.EVENT_LBUTTONDOWN:
        print(trackWindow is not None)

    # boundary 박스 지정
    if inputMode:

        # 박스 지정 시작
        if event == cv2.EVENT_LBUTTONDOWN:
            rectangle = True
            col, row = x, y

        # 박스 크기 확장
        elif event == cv2.EVENT_MOUSEMOVE:
            if rectangle:
                frame = frame2.copy()
                cv2.rectangle(frame, (col, row), (x, y), (255, 255, 255), 2)
                cv2.imshow('frame', frame)

        # 박스 확정
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

    # 박스 지정이 끝나면 색상을 문질문질
    elif inputMode2:
        
        # 색상 지정 시작
        if not clFlag and event == cv2.EVENT_LBUTTONDOWN:
            clFlag = True
            print('downevent')

        # 색상 문질문질
        if clFlag and event == cv2.EVENT_MOUSEMOVE:
            poip = poip + 1
            ku = frame[y, x].copy()
            # 문질문질해가며 색상값을 다 더함
            colorSum = [colorSum[0] + ku[0],
                        colorSum[1] + ku[1], colorSum[2] + ku[2]]

        # 색상 설정 마무리
        if poip != 0 and event == cv2.EVENT_LBUTTONUP:
            print('upevent')
            clFlag = False
            # 다 더한값의 평균을 구함
            mgr = [colorSum[0]/poip, colorSum[1]/poip, colorSum[2]/poip]

            print('mgr: ' + str(mgr))

            # 크기 반대순으로 솔팅
            sortedValue = sorted(
                [0, 1, 2], key=lambda x: mgr[x], reverse=False)

            # color 값을 세 색의 합으로 나누어주기 위해서 선언
            colorContectValue = mgr[0] + mgr[1] + mgr[2]

            print(sortedValue)
            los = []
            ups = []

            # 값의 중요도에 따라 가중치 적용
            # 정확한게 아니라 수정해야할 수 도있음
            for k in sortedValue:
                colorDifferenceValues[k] = mgr[k] * \
                    colorDifferenceValue / colorContectValue

            # 가중치를 고려하여 0 - 255 사이의 값으로 나타내기
            for i in range(0, 3):
                los.append(mgr[i] - colorDifferenceValues[i]
                           if mgr[i] > colorDifferenceValues[i] else 0)
                ups.append(mgr[i] + colorDifferenceValues[i]
                           if mgr[i] < 255-colorDifferenceValues[i] else 255)

            # 색상의 경계 설정 완료
            boundaries[0] = (los, ups)
            lower = np.array(los, dtype="uint8")
            upper = np.array(ups, dtype="uint8")

            main.eval('tk::PlaceWindow %s center' % main.winfo_pathname(main.winfo_id()))
            # color picker 띄워줌
            main.mainloop()

    return

# 시작
def start():
    global frame, frame2, inputMode, inputMode, trackWindow, roi_hist, roi, boundaries, blank_image3, s, h, w, upper, lower
    global fcc, out, tracker_initation, myTrackerType

    try:
        # 비디오 가져옴
        cap = cv2.VideoCapture('video/sampleVideo.mp4')
        # 크기설정
        cap.set(3, 480)
        cap.set(4, 320)
        wx = int(cap.get(3))
        hx = int(cap.get(4))

        # output 출력을 위한 VideoWriter 초기화
        out = cv2.VideoWriter('output.mp4', fcc, 20, (wx, hx))

    except:
        print('Cam Failed')
        return

    # 박스 지정을위해
    ret, frame = cap.read()

    # 이름설정
    cv2.namedWindow('frame')
    # 마우스 이벤트 리스너 생성
    cv2.setMouseCallback('frame', onMouse, param=(frame, frame2))

    # lower 색상과 upper 색상 초기화
    lower, upper = boundaries[0]
    lower = np.array(lower, dtype="uint8")
    upper = np.array(upper, dtype="uint8")

    # 재생시작
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        h, w = frame.shape[:2]

        # 초기 값은 0,0,0 부터 255,255,255 이므로 현재 마스크는 의미가 없지만,
        # 마우스 이벤트에서 색상을 입력받은 후 부터는 의미를 가지게됨
        mask = cv2.inRange(frame, lower, upper) 
        
        # subtract 메서드 사용을 위해 BGR포멧으로 변경해줌
        # 왠진 모름 오류가 나더라
        mask2 = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        resf = cv2.subtract(frame, mask2)

        # 바꾸고자 하는 색상을 가진 video와 같은 크기의 이미지
        blank_image = np.zeros((h, w, 3), np.uint8)
        blank_image[:] = cvt2Colors

        # 색상 변경을 위해 HSV포맷으로 변경
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_BGR2HSV)
        
        # 이건 검은 이미지가 필요해서 만듬
        blank_image2 = np.zeros((h, w, 3), np.uint8)
        blank_image2[:] = (255, 255, 255)

        # hsv색상에서 명도만 뽑아와서 이미지에 덧씌움
        ppap = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2HSV)
        blank_image[:, :, 2] = ppap[:, :, 2]
        blank_image = cv2.cvtColor(blank_image, cv2.COLOR_HSV2BGR)

        # 마우스 이벤트 후
        if s:
            # 비트 연산을 통해 색상 추출 및 색상 변경
            mask3 = cv2.bitwise_and(blank_image3, mask2)
            resf = cv2.subtract(frame, mask3)

            # mask로 사용하기 위해 다시 GRAY포맷으로 변경
            mask3 = cv2.cvtColor(mask3, cv2.COLOR_BGR2GRAY)
            frame2 = cv2.bitwise_or(resf, blank_image, mask=mask3)
            
            # 최종 결과 이미지
            frame2 = cv2.bitwise_or(resf, frame2)

        # 마우스 이벤트 전
        else:
            # 컨버팅 과정을 보고싶어서 넣어놓은것
            # 마우스 이벤트 전에는 어차피 frame2가 아닌 frame을 출력하기 때문에 무의미하다. 
            frame2 = cv2.bitwise_or(resf, blank_image, mask=mask)
            frame2 = cv2.bitwise_or(resf, frame2)

        # mouse 이벤트 후
        if trackWindow is not None:

            # 최초로 진입했다면 tracker가 추적하기 위한 boundary 박스를 전달
            if tracker_initation:
                tracker = tracker_create(myTrackerType)
                ret =tracker.init(frame2, trackWindow)
                tracker_initation = False
            
            # boundary 박스를 따라서 추적을 진행
            else:
                ret, trackWindow = tracker.update(frame)
            
            # 변수명이 너무 길어서 만든거
            a = trackWindow

            # boundary 박스의 모든 점 위치를 찍어ㅈ무
            pts = [
                [a[0], a[1]],
                [a[0] + a[2], a[1]],
                [a[0]+a[2], a[1]+a[3]],
                [a[0], a[1]+a[3]]]
            pts = np.int0(pts)
            copyPts = pts.copy()

            center = [0, 0]
            #print(copyPts)

            # tracking 알고리즘 찾아보기 전에
            # 약간이나마 tracking을 최적화 시키기 위해 중심좌표를 계산하여 가장 거리가 먼 
            # 좌표를 한 점으로 가지는 정사각형으로 boundary 를 변환해주는 과정
            # 근데 이제 다른 알고리즘 찾아서 쓸모가 없다
            for gp in copyPts:
                center[0] = center[0] + gp[0]
                center[1] = center[1] + gp[1]

            center[0] = center[0]/4
            center[1] = center[1]/4
        
            diff = 0

            # 가장 큰 값을 구함
            for b in copyPts:
                dif_ = (b[0] - center[0])**2 + (b[1] - center[1])**2
                if(dif_ > diff):
                    diff = dif_

            diff = math.sqrt(diff)/math.sqrt(2)
            #print(diff)
            copyPts[0][0] = center[0] + diff
            copyPts[0][1] = center[1] + diff
            copyPts[1][0] = center[0] + diff
            copyPts[1][1] = center[1] - diff
            copyPts[2][0] = center[0] - diff
            copyPts[2][1] = center[1] - diff
            copyPts[3][0] = center[0] - diff
            copyPts[3][1] = center[1] + diff
            #print(ret)

            # boundary 박스가 화면 밖으로 나가는 것을 막기 위해 만듬
            for (i, asdf) in enumerate(pts):
                if asdf[0] >= w:
                    copyPts[i][0] = w
                elif asdf[0] <= 0:
                    copyPts[i][0] = 0
                if asdf[1] >= h:
                    copyPts[i][1] = h
                elif asdf[1] <= 0:
                    copyPts[i][1] = 0

            blank_image3[:] = (0, 0, 0)
            blank_image3[np.min(copyPts[:, 1]):np.max(copyPts[:, 1]), np.min(
                copyPts[:, 0]):np.max(copyPts[:, 0])] = (255, 255, 255)
            
            # 이 프레임은 출력용 프레임으로 저장용 프레임에 바운딩 박스를 적용시키지 않기 위해서 미리 얕은 복사를 진행
            writeFrame = frame2.copy()
            cv2.putText(writeFrame, myTrackerType, (0,50), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255,0,0), 2)

            # 박스로 객체를 추적하는것을 보여주기 위함
            cv2.polylines(writeFrame, [copyPts], True, (0, 255, 0), 2)

        # 마우스 이벤트 발생 후
        if s:
            # 출력용 프레임
            cv2.imshow('frame', writeFrame)

            # 저장용 프레임
            out.write(frame2)

        # 마우스 이벤트 발생 전
        else:
            # 출력용 프레임
            cv2.imshow('frame', frame)

        # 키 이벤트를 대기
        k = cv2.waitKey(10) & 0xFF

        if k == 27:
            break

        # 마우스 이벤트를 발생시키기 위한 트리거 장치
        if k == ord('i'):
            print('추적한 영역을 지정하고 아무키나 누르세요')
            inputMode = True

            # 내부 주소값까지 복사하면 안되므로 깊은복사 사용
            frame2 = copy.copy(frame)

            # 마우스 이벤트 동작중
            while inputMode or inputMode2:
                cv2.imshow('frame', frame)
                cv2.waitKey(0)

        # 종료 트리거이며 쌓여왔던 프레임을 저장
        if k == ord('q'):
            print('종료되었습니다.')
            out.release()
            break

    cap.release()

    #프레임 저장
    out.release()
    cv2.destroyAllWindows()


# 시작
if __name__ == '__main__':
    # GUI 데이터
    main = Tk()
    main.title('SVC')
    main.protocol("WM_DELETE_WINDOW", on_closing)
    main.geometry('200x200')

    # 옵션 메뉴에 들어갈 values
    variable = StringVar(main)
    variable.set(tracker_types[0])

    # 옵션 메뉴
    opt = OptionMenu(main, variable, *tracker_types)
    opt.config(width = 90, font=("Helvetica", 12))
    opt.pack()

    # colorPicker를 호출하기 위한 버튼
    Button(main, text = "color", command=getColor).pack()

    # 적용 버튼
    Button(main, text = "적용", command=on_closing).pack()

    # 시작
    start()