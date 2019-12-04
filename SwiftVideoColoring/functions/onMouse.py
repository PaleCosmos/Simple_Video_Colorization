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
            ku = frame[y, x].copy()
            colorSum = [colorSum[0] + ku[0],
                        colorSum[1] + ku[1], colorSum[2] + ku[2]]

        if poip != 0 and event == cv2.EVENT_LBUTTONUP:
            print('upevent')
            clFlag = False

            mgr = [colorSum[0]/poip, colorSum[1]/poip, colorSum[2]/poip]

            print('mgr: ' + str(mgr))

            sortedValue = sorted(
                [0, 1, 2], key=lambda x: mgr[x], reverse=False)

            colorContectValue = mgr[0] + mgr[1] + mgr[2]

            print(sortedValue)
            los = []
            ups = []

            for k in sortedValue:
                colorDifferenceValues[k] = mgr[k] * \
                    colorDifferenceValue / colorContectValue

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

