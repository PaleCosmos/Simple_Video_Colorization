import cv2
import sys
from functions.onMouse import onMouse as onmouse

colorDifferenceValue = 255
colorDifferenceValues = [colorDifferenceValue,
                         colorDifferenceValue, colorDifferenceValue]

boundaries = [
    ([0, 0, 0], [255, 255, 255])
]

cvt2Colors = (255, 0, 0)

fcc = cv2.VideoWriter_fourcc(*'mp4v')
out = None

tracker = cv2.TrackerMIL_create()
cvt2Colors = (255, 0, 0)

video = cv2.VideoCapture('sampleVideo.mp4')

if not video.isOpened():
    print("Could not open video")
    sys.exit()

ok, frame = video.read()
if not ok:
    print('Cannot read video file')
    sys.exit()

bbox = cv2.selectROI(frame, False)

ok = tracker.init(frame, bbox)
 
while True:
    # Read a new frame
    ok, frame = video.read()
    if not ok:
        break
        
    # Start timer
    timer = cv2.getTickCount()

    # Update tracker
    ok, bbox = tracker.update(frame)

    # Calculate Frames per second (FPS)
    fps = cv2.getTickFrequency() / (cv2.getTickCount() - timer);

    # Draw bounding box
    if ok:
        # Tracking success
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)
        print(bbox)
    else :
        # Tracking failure
        cv2.putText(frame, "Tracking failure detected", (100,80), cv2.FONT_HERSHEY_SIMPLEX, 0.75,(0,0,255),2)

    # Display tracker type on frame
    #cv2.putText(frame, " Tracker", (100,20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50),2);
    
    # Display FPS on frame
    #cv2.putText(frame, "FPS : " + str(int(fps)), (100,50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (50,170,50), 2);

    # Display result
    cv2.imshow("Tracking", frame)

    # Exit if ESC pressed
    k = cv2.waitKey(1) & 0xff
    if k == 27 : break