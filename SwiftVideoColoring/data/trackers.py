import cv2

# 다양한 tracker의 종류들
# 아직 모든 알고리즘을 적용해 본것이 아니기 때문에
# 한번씩 다 적용해보고 가장 알맞은 알고리즘을 찾아야함
def tracker_create(tracker_type):
    if tracker_type == 'BOOSTING':
        tracker = cv2.TrackerBoosting_create()
    if tracker_type == 'MIL':
        tracker = cv2.TrackerMIL_create()
    if tracker_type == 'KCF':
        tracker = cv2.TrackerKCF_create()
    if tracker_type == 'TLD':
        tracker = cv2.TrackerTLD_create()
    if tracker_type == 'MEDIANFLOW':
        tracker = cv2.TrackerMedianFlow_create()
    if tracker_type == 'CSRT':
        tracker = cv2.TrackerCSRT_create()
    if tracker_type == 'MOSSE':
        tracker = cv2.TrackerMOSSE_create()
    return tracker