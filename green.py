import numpy as np
import cv2
from collections import deque
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
                help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
                help="max buffer size")
args = vars(ap.parse_args())

cap = cv2.VideoCapture(0)
fps = cap.get(cv2.CAP_PROP_FPS)

greenLower = (88, 43, 46)                 # 29, 86, 6
greenUpper = (124, 255, 255)                    # 64, 255, 255
pts = deque(maxlen=args["buffer"])      # 双向队列

while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    # Display the resulting frame
    # frame = cv2.medianBlur(frame, 5)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, greenLower, greenUpper)       # 将低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255
    green_mask = cv2.erode(green_mask, None, iterations=2)      # 腐蚀
    green_mask = cv2.dilate(green_mask, None, iterations=2)     # 膨胀
    cnts, hie = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # 查找检测物体的轮廓，返回三个参数
    center = None
    if len(cnts) > 0:
        c = max(cnts, key=cv2.contourArea)
        # ((x, y), radius) = cv2.minEnclosingCircle(c)        # 使用迭代算法查找包含2D点集的最小区域的圆
        # a1 = cv2.minAreaRect(c)
        x, y, w, h = cv2.boundingRect(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))       # 圆心

        if w > 10 or h > 10:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # if radius > 10:
        #     cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)      # 画框
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)       # 画圆心
    pts.appendleft(center)      # 在队首加入一个元素
    for i in range(1, len(pts)):
        if pts[i - 1] is None or pts[i] is None:
            continue

        thickness = int(np.sqrt(args["buffer"] / float(i + 1)) * 2.5)
        # cv2.line(frame, pts[i - 1], pts[i], (0, 0, 255), thickness)
        cv2.line(frame, pts[i], pts[i - 1], (0, 0, 255), thickness)
        distance = np.linalg.norm(np.array(pts[i]) - np.array(pts[i - 1]))
        velocity = distance * fps
        print(velocity)
        '''
        if velocity < 40:
            print("over")
        '''

    cv2.imshow("xx", green_mask)
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()