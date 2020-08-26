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

cap = cv2.VideoCapture(1, cv2.CAP_DSHOW)
# fps = cap.get(cv2.CAP_PROP_FPS)

"""
blue:
88, 43, 46
124, 255, 255

green:
29, 86, 6
64, 255, 255
"""
greenLower = (88, 43, 46)                 # 29, 86, 6
greenUpper = (124, 255, 255)                    # 64, 255, 255
pts = deque(maxlen=args["buffer"])      # 双向队列


def contrast_image_correction(img):
    """
    contrast image correction论文的python复现，HDR技术
    :param img: cv2.imread读取的图片数据
    :return: 返回的HDR校正后的图片数据
    """
    img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    mv = cv2.split(img_yuv)
    img_y = mv[0].copy();

    # temp = img_y
    temp = cv2.bilateralFilter(mv[0], 9, 50, 50)
    # for i in range(len(img)):
    #     for j in range(len(img[0])):
    #         exp = np.power(2, (128 - (255 - temp[i][j])) / 128.0)
    #         temp[i][j] = int(255 * np.power(img_y[i][j] / 255.0, exp))
    #         # print(exp.dtype)
    # print(temp.dtype)
    exp = np.power(2, (128.0 - (255 - temp).astype(np.float32)) / 128.0)
    temp = (255 * np.power(img_y.flatten() / 255.0, exp.flatten())).astype(np.uint8)
    temp = temp.reshape((img_y.shape))

    dst = img.copy()

    img_y[img_y == 0] = 1
    for k in range(3):
        val = temp / img_y
        val1 = img[:, :, k].astype(np.int32) + img_y.astype(np.int32)
        val2 = (val * val1 + img[:, :, k] - img_y) / 2
        dst[:, :, k] = val2.astype(np.int32)

    # for i in range(len(img)):
    #     for j in range(len(img[0])):
    #         if (img_y[i][j] == 0):
    #             dst[i, j, :] = 0
    #         else:
    #             for k in range(3):
    #                 val = temp[i, j]/img_y[i, j]
    #                 val1 = int(img[i, j, k]) + int(img_y[i, j])
    #                 val2 = (val * val1+ img[i, j, k] - img_y[i, j]) / 2
    #                 dst[i, j, k] = int(val2)
    #             """
    #             BUG:直接用下面计算方法会导致值溢出，导致计算结果不正确
    #             """
    # dst[i, j, 0] = (temp[i, j] * (img[i, j, 0] + img_y[i, j]) / img_y[i, j] + img[i, j, 0] - img_y[
    #     i, j]) / 2
    # dst[i, j, 1] = (temp[i, j] * (img[i, j, 1] + img_y[i, j]) / img_y[i, j] + img[i, j, 1] - img_y[
    #     i, j]) / 2
    # dst[i, j, 2] = (temp[i, j] * (img[i, j, 2] + img_y[i, j]) / img_y[i, j] + img[i, j, 2] - img_y[
    #     i, j]) / 2

    return dst

def unevenLightCompensate(img, blockSize):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    average = np.mean(gray)

    rows_new = int(np.ceil(gray.shape[0] / blockSize))
    cols_new = int(np.ceil(gray.shape[1] / blockSize))

    blockImage = np.zeros((rows_new, cols_new), dtype=np.float32)
    for r in range(rows_new):
        for c in range(cols_new):
            rowmin = r * blockSize
            rowmax = (r + 1) * blockSize
            if (rowmax > gray.shape[0]):
                rowmax = gray.shape[0]
            colmin = c * blockSize
            colmax = (c + 1) * blockSize
            if (colmax > gray.shape[1]):
                colmax = gray.shape[1]

            imageROI = gray[rowmin:rowmax, colmin:colmax]
            temaver = np.mean(imageROI)
            blockImage[r, c] = temaver

    blockImage = blockImage - average
    blockImage2 = cv2.resize(blockImage, (gray.shape[1], gray.shape[0]), interpolation=cv2.INTER_CUBIC)
    gray2 = gray.astype(np.float32)
    dst = gray2 - blockImage2
    dst = dst.astype(np.uint8)
    dst = cv2.GaussianBlur(dst, (3, 3), 0)
    dst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    return dst

while True:
    ret, frame = cap.read()
    # Capture frame-by-frame
    # frame = cv2.imread("1.jpg")
    # Display the resulting frame
    # frame = cv2.medianBlur(frame, 5)
    # frame = contrast_image_correction(frame)        # 模仿HDR校正
    # frame = unevenLightCompensate(frame, 16)
    # frame = np.concatenate([frame, dst], axis=1)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    green_mask = cv2.inRange(hsv, greenLower, greenUpper)       # 将低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)  # 开操作
    green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)  # 闭操作

    # green_mask = cv2.erode(green_mask, None, iterations=2)      # 腐蚀
    # green_mask = cv2.dilate(green_mask, None, iterations=2)     # 膨胀
    cnts, hie = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)     # 查找检测物体的轮廓，返回三个参数
    center = None
    if len(cnts) > 4:
        # c = max(cnts, key=cv2.contourArea)
        cnts.sort(key=cv2.contourArea, reverse=True)
        # ((x, y), radius) = cv2.minEnclosingCircle(c)        # 使用迭代算法查找包含2D点集的最小区域的圆
        # a1 = cv2.minAreaRect(c)
        x1, y1, w1, h1 = cv2.boundingRect(cnts[0])
        x2, y2, w2, h2 = cv2.boundingRect(cnts[1])
        x3, y3, w3, h3 = cv2.boundingRect(cnts[2])
        x4, y4, w4, h4 = cv2.boundingRect(cnts[3])
        # M1 = cv2.moments(cnts[0])
        # M2 = cv2.moments(cnts[1])
        # M3 = cv2.moments(cnts[2])
        # M4 = cv2.moments(cnts[3])
        # center1 = (int(M1["m10"] / M1["m00"]), int(M1["m01"] / M1["m00"]))       # 圆心
        # center2 = (int(M2["m10"] / M2["m00"]), int(M2["m01"] / M2["m00"]))
        # center3 = (int(M3["m10"] / M3["m00"]), int(M3["m01"] / M3["m00"]))
        # center4 = (int(M4["m10"] / M4["m00"]), int(M4["m01"] / M4["m00"]))

        center1 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
        center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
        center3 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
        center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))

        center = (int((center1[0]+center2[0]+center3[0]+center4[0])/4), int((center1[1]+center2[1]+center3[1]+center4[1])/4))
        cv2.circle(frame, center, 5, (0, 0, 255), -1)
        # cv2.circle(frame, center, 5, (255, 0, 0), -1)

        # if w > 10 or h > 10:
        #     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)

        # if radius > 10:
        #     cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)      # 画框
        #     cv2.circle(frame, center, 5, (0, 0, 255), -1)       # 画圆心
    pts.appendleft(center)      # 在队首加入一个元素

    cv2.imshow("xx", green_mask)
    cv2.imshow("Frame", frame)
    k = cv2.waitKey(1)
    if k==27:
        break

cap.release()
cv2.destroyAllWindows()
