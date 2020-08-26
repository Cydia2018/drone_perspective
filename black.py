import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # 或传入0，使用摄像头

while (True):

    # 读取一帧
    _, frame = cap.read()

    # 把 BGR 转为 HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # HSV中黑色范围
    lower_blue = np.array([0, 0, 0])
    upper_blue = np.array([180, 255, 46])

    # 获得黑色区域的mask
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # 和原始图片进行and操作，获得黑色区域
    res = cv2.bitwise_and(frame, frame, mask=mask)

    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    cv2.namedWindow('mask', cv2.WINDOW_NORMAL)
    cv2.imshow('mask', mask)
    cv2.namedWindow('res', cv2.WINDOW_NORMAL)
    cv2.imshow('res', res)

    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()