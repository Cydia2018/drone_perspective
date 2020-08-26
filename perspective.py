from djitellopy import Tello
import numpy as np
import cv2
import time
import torch
import threading
import pygame
from pygame.locals import *
from agents.ddpg import Actor

S = 30
Lower = (35, 43, 46)    # 26--34
Upper = (79, 255, 255)

def get_dist(x, y):
    dist = np.sqrt((np.square(np.array([x, y]) - np.array([480, 360]))).sum())
    return dist

class TelloControl(object):
    """ Maintains the Tello display and moves it through the keyboard keys.
        Press escape key to quit.
        The controls are:
            - T: Takeoff
            - L: Land
            - Arrow keys: Forward, backward, left and right.
            - A and D: Counter clockwise and clockwise rotations
            - W and S: Up and down.
    """

    def __init__(self):
        # Init pygame
        pygame.init()

        # Creat pygame window
        pygame.display.set_caption("control stream")
        self.screen = pygame.display.set_mode([960, 720])
        # self.screen = pygame.display.set_mode([416, 416])

        # Init Tello object that interacts with the Tello drone
        self.tello = Tello()
        self.actor = Actor(30.0)
        self.actor.load_state_dict(torch.load("agents/actor.pth"))

        # Drone velocities between -100~100
        self.for_back_velocity = 0
        self.left_right_velocity = 0
        self.up_down_velocity = 0
        self.yaw_velocity = 0
        self.speed = 60

        self.send_rc_control = False

        # create update timer
        pygame.time.set_timer(USEREVENT + 1, 50)
        # pygame.time.set_timer(USEREVENT + 2, 250)

        # Run thread to find box in frame
        # print('init done')

    def run(self):
        if not self.tello.connect():
            print("Tello not connected")
            return
        if not self.tello.set_speed(self.speed):
            print("Not set speed to lowest possible")
            return

        self.tello.streamoff()
        self.tello.streamon()
        cap = self.tello.get_frame_read()

        should_stop = False
        dstImage = None
        # print('loop started')

        while not should_stop:
            frame = cap.frame            # 摄像头读取一帧
            for event in pygame.event.get():
                if event.type == USEREVENT + 1:
                    self.update()
                elif event.type == QUIT:
                    should_stop = True
                elif event.type == KEYDOWN:
                    if event.key == K_ESCAPE:
                        should_stop = True
                    else:
                        self.keydown(event.key)
                elif event.type == KEYUP:
                    self.keyup(event.key)

            self.screen.fill([0, 0, 0])

            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            green_mask = cv2.inRange(hsv, Lower, Upper)  # 将低于lower_red和高于upper_red的部分分别变成0，lower_red～upper_red之间的值变成255
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel)  # 开操作
            green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel)  # 闭操作
            cnts, hie = cv2.findContours(green_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            if len(cnts) > 3:
                cnts.sort(key=cv2.contourArea, reverse=True)
                x1, y1, w1, h1 = cv2.boundingRect(cnts[0])
                x2, y2, w2, h2 = cv2.boundingRect(cnts[1])
                x3, y3, w3, h3 = cv2.boundingRect(cnts[2])
                x4, y4, w4, h4 = cv2.boundingRect(cnts[3])

                x1 = x1 + 1
                y1 = y1 + 1
                x2 = x2 + 1
                y2 = y2 + 1
                x3 = x3 + 1
                y3 = y3 + 1
                x4 = x4 + 1
                y4 = y4 + 1

                if min(x1*y1, x2*y2, x3*y3, x4*y4) == x1*y1:
                    center1 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                    part1 = (int(x1 + w1), int(y1 + h1))
                    if min(abs(x2 - x1), abs(x3 - x1), abs(x4 - x1)) == abs(x2 - x1):  # 3
                        center3 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                        part3 = (int(x2 + w2), int(y2))
                        if min(abs(y3 - y1), abs(y4 - y1)) == abs(y3 - y1):  # 2
                            center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part2 = (int(x3), int(y3 + h3))
                            center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part4 = (int(x4), int(y4))
                        else:
                            center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part2 = (int(x4), int(y4 + h4))
                            center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part4 = (int(x3), int(y3))
                    elif min(abs(x2 - x1), abs(x3 - x1), abs(x4 - x1)) == abs(x3 - x1):
                        center3 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                        part3 = (int(x3 + w3), int(y3))
                        if min(abs(y2 - y1), abs(y4 - y1)) == abs(y2 - y1):  # 2
                            center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part2 = (int(x2), int(y2 + h2))
                            center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part4 = (int(x4), int(y4))
                        else:
                            center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part2 = (int(x4), int(y4 + h4))
                            center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part4 = (int(x2), int(y2))
                    else:
                        center3 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                        part3 = (int(x4 + w4), int(y4))
                        if min(abs(y2 - y1), abs(y3 - y1)) == abs(y2 - y1):  # 2
                            center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part2 = (int(x2), int(y2 + h2))
                            center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part4 = (int(x3), int(y3))
                        else:
                            center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part2 = (int(x3), int(y3 + h3))
                            center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part4 = (int(x2), int(y2))
                elif min(x1*y1, x2*y2, x3*y3, x4*y4) == x2*y2:
                    center1 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                    part1 = (int(x2 + w2), int(y2 + h2))
                    if min(abs(x1 - x2), abs(x3 - x2), abs(x4 - x2)) == abs(x1 - x2):  # 3
                        center3 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                        part3 = (int(x1 + w1), int(y1))
                        if min(abs(y3 - y2), abs(y4 - y2)) == abs(y3 - y2):  # 2
                            center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part2 = (int(x3), int(y3 + h3))
                            center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part4 = (int(x4), int(y4))
                        else:
                            center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part2 = (int(x4), int(y4 + h4))
                            center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part4 = (int(x3), int(y3))
                    elif min(abs(x1 - x2), abs(x3 - x2), abs(x4 - x2)) == abs(x3 - x2):
                        center3 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                        part3 = (int(x3 + w3), int(y3))
                        if min(abs(y1 - y2), abs(y4 - y2)) == abs(y1 - y2):  # 2
                            center2 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part2 = (int(x1), int(y1 + h1))
                            center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part4 = (int(x4), int(y4))
                        else:
                            center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part2 = (int(x4), int(y4 + h4))
                            center4 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part4 = (int(x1), int(y1))
                    else:
                        center3 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                        part3 = (int(x4 + w4), int(y4))
                        if min(abs(y1 - y2), abs(y3 - y2)) == abs(y1 - y2):  # 2
                            center2 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part2 = (int(x1), int(y1 + h1))
                            center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part4 = (int(x3), int(y3))
                        else:
                            center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part2 = (int(x3), int(y3 + h3))
                            center4 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part4 = (int(x1), int(y1))
                elif min(x1*y1, x2*y2, x3*y3, x4*y4) == x3*y3:
                    center1 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                    part1 = (int(x3 + w3), int(y3 + h3))
                    if min(abs(x1 - x3), abs(x2 - x3), abs(x4 - x3)) == abs(x1 - x3):  # 3
                        center3 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                        part3 = (int(x1 + w1), int(y1))
                        if min(abs(y2 - y3), abs(y4 - y3)) == abs(y2 - y3):  # 2
                            center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part2 = (int(x2), int(y2 + h2))
                            center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part4 = (int(x4), int(y4))
                        else:
                            center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part2 = (int(x4), int(y4 + h4))
                            center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part4 = (int(x2), int(y2))
                    elif min(abs(x1 - x3), abs(x2 - x3), abs(x4 - x3)) == abs(x2 - x3):
                        center3 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                        part3 = (int(x2 + w2), int(y2))
                        if min(abs(y1 - y3), abs(y4 - y3)) == abs(y1 - y3):  # 2
                            center2 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part2 = (int(x1), int(y1 + h1))
                            center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part4 = (int(x4), int(y4))
                        else:
                            center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                            part2 = (int(x4), int(y4 + h4))
                            center4 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part4 = (int(x1), int(y1))
                    else:
                        center3 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                        part3 = (int(x4 + w4), int(y4))
                        if min(abs(y1 - y3), abs(y2 - y3)) == abs(y1 - y3):  # 2
                            center2 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part2 = (int(x1), int(y1 + h1))
                            center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part4 = (int(x2), int(y2))
                        else:
                            center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part2 = (int(x2), int(y2 + h2))
                            center4 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part4 = (int(x1), int(y1))
                else:
                    center1 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                    part1 = (int(x4 + w4), int(y4 + h4))
                    if min(abs(x1 - x4), abs(x2 - x4), abs(x3 - x4)) == abs(x1 - x4):  # 3
                        center3 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                        part3 = (int(x1 + w1), int(y1))
                        if min(abs(y2 - y4), abs(y3 - y4)) == abs(y2 - y4):  # 2
                            center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part2 = (int(x2), int(y2 + h2))
                            center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part4 = (int(x3), int(y3))
                        else:
                            center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part2 = (int(x3), int(y3 + h3))
                            center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part4 = (int(x2), int(y2))
                    elif min(abs(x1 - x4), abs(x2 - x4), abs(x3 - x4)) == abs(x2 - x4):
                        center3 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                        part3 = (int(x2 + w2), int(y2))
                        if min(abs(y1 - y4), abs(y3 - y4)) == abs(y1 - y4):  # 2
                            center2 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part2 = (int(x1), int(y1 + h1))
                            center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part4 = (int(x3), int(y3))
                        else:
                            center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                            part2 = (int(x3), int(y3 + h3))
                            center4 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part4 = (int(x1), int(y1))
                    else:
                        center3 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                        part3 = (int(x3 + w3), int(y3))
                        if min(abs(y1 - y4), abs(y2 - y4)) == abs(y1 - y4):  # 2
                            center2 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part2 = (int(x1), int(y1 + h1))
                            center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part4 = (int(x2), int(y2))
                        else:
                            center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                            part2 = (int(x2), int(y2 + h2))
                            center4 = (int(x1 + w1 / 2), int(y1 + h1 / 2))
                            part4 = (int(x1), int(y1))

                # if min(abs(x2 - x1), abs(x3 - x1), abs(x4 - x1)) == abs(x2 - x1):  # 3
                #     center3 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                #     part3 = (int(x2 + w2), int(y2))
                #     if min(abs(y3 - y1), abs(y4 - y1)) == abs(y3 - y1):  # 2
                #         center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                #         part2 = (int(x3), int(y3 + h3))
                #         center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                #         part4 = (int(x4), int(y4))
                #     else:
                #         center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                #         part2 = (int(x4), int(y4 + h4))
                #         center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                #         part4 = (int(x3), int(y3))
                # elif min(abs(x2 - x1), abs(x3 - x1), abs(x4 - x1)) == abs(x3 - x1):
                #     center3 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                #     part3 = (int(x3 + w3), int(y3))
                #     if min(abs(y2 - y1), abs(y4 - y1)) == abs(y2 - y1):  # 2
                #         center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                #         part2 = (int(x2), int(y2 + h2))
                #         center4 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                #         part4 = (int(x4), int(y4))
                #     else:
                #         center2 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                #         part2 = (int(x4), int(y4 + h4))
                #         center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                #         part4 = (int(x2), int(y2))
                # else:
                #     center3 = (int(x4 + w4 / 2), int(y4 + h4 / 2))
                #     part3 = (int(x4 + w4), int(y4))
                #     if min(abs(y2 - y1), abs(y3 - y1)) == abs(y2 - y1):  # 2
                #         center2 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                #         part2 = (int(x2), int(y2 + h2))
                #         center4 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                #         part4 = (int(x3), int(y3))
                #     else:
                #         center2 = (int(x3 + w3 / 2), int(y3 + h3 / 2))
                #         part2 = (int(x3), int(y3 + h3))
                #         center4 = (int(x2 + w2 / 2), int(y2 + h2 / 2))
                #         part4 = (int(x2), int(y2))

                center = (int((center1[0] + center2[0] + center3[0] + center4[0]) / 4),
                          int((center1[1] + center2[1] + center3[1] + center4[1]) / 4))

                # pts = np.float32(
                #     [[part1[0], part1[1]], [part2[0], part2[1]], [part3[0], part3[1]], [part4[0], part4[1]]])
                pts = np.float32([[center1[0], center1[1]], [center2[0], center2[1]], [center3[0], center3[1]],
                                  [center4[0], center4[1]]])
                # dst = np.float32([[0, 0], [608, 0], [0, 608], [608, 608]])
                dst = np.float32([[0, 0], [640, 0], [0, 480], [640, 480]])
                # srcPic = frame[part1[1]:part3[1], part1[0]:part2[0]]

                M = cv2.getPerspectiveTransform(pts, dst)
                # dstImage = cv2.warpPerspective(frame, M, (608, 608))
                dstImage = cv2.warpPerspective(frame, M, (640, 480))

                w = max(abs(center1[0]-center2[0]), abs(center1[0]-center3[0]), abs(center1[0]-center4[0]))
                h = max(abs(center1[1] - center2[1]), abs(center1[1] - center3[1]), abs(center1[1] - center4[1]))
                # delta = min(abs(center2[1]-center1[1]), abs(center3[1]-center1[1]), abs(center4[1]-center1[1]))
                # temp_a = center2[1]-center1[1]
                # temp_b = center3[1]-center1[1]
                # temp_c = center4[1]-center1[1]
                # delta = min(abs(temp_a), abs(temp_b), abs(temp_c))
                # flag = bool(temp_a * temp_b * temp_c > 0)   # 左偏

                percent = (w*h / 691200.) * 100.0
                cnt = torch.Tensor(center)
                actions = self.actor(cnt)
                actions = actions.cpu().data.numpy()
                # actions = speedControlLinear(center)

                # done = bool(get_dist(center[0], center[1]) < 60 and (percent > 40)) and (delta < 10)
                done = bool(get_dist(center[0], center[1]) < 60 and (percent > 40))
                if not done:
                    self.yaw_velocity = -int(actions[0])     # 强化学习需要加"-"
                    self.up_down_velocity = int(actions[1])
                    if percent < 40:
                        self.for_back_velocity = 13
                        # if delta > 10:
                        #     # self.left_right_velocity = 10
                        #     # self.yaw_velocity = -10
                        #     if flag:
                        #         self.left_right_velocity = 10
                        #         self.yaw_velocity = -10
                        #     else:
                        #         self.left_right_velocity = -10
                        #         self.yaw_velocity = 10
                    # cv2.circle(frame, center, 5, (255, 0, 0), -1)
                else:
                    self.yaw_velocity = 0
                    self.up_down_velocity = 0
                    self.for_back_velocity = 0
            else:
                self.for_back_velocity = -15
                self.yaw_velocity = -self.yaw_velocity/2
                self.up_down_velocity = 0
                # self.up_down_velocity = -self.up_down_velocity/2

            # frame = cv2.circle(frame, (480, 360), 5,(0, 0, 255),-1)
            cv2.imshow("xx", green_mask)
            if dstImage is not None:
                cv2.imshow("dst", dstImage)
            k = cv2.waitKey(1)
            if k == 27:
                break

            frame = np.rot90(frame)
            # frame = np.rot90(dstImage)
            frame = np.flipud(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = pygame.surfarray.make_surface(frame)
            self.screen.blit(frame, (0, 0))
            pygame.display.update()

            # time.sleep(0.1)

        # Call it always before finishing. I deallocate resources.
        self.tello.end()

    def keydown(self, key):
        """ Update velocities based on key pressed
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP:  # set forward velocity      ↑键前进
            self.for_back_velocity = S
        elif key == pygame.K_DOWN:  # set backward velocity     ↓键后退
            self.for_back_velocity = -S
        elif key == pygame.K_LEFT:  # set left velocity         ←键左移
            self.left_right_velocity = -S
        elif key == pygame.K_RIGHT:  # set right velocity       →键右移
            self.left_right_velocity = S
        elif key == pygame.K_w:  # set up velocity              w键上升
            self.up_down_velocity = S
        elif key == pygame.K_s:  # set down velocity            s键下降
            self.up_down_velocity = -S
        elif key == pygame.K_a:  # set yaw counter clockwise velocity       a键逆时针
            self.yaw_velocity = -S
        elif key == pygame.K_d:  # set yaw clockwise velocity       d键顺时针
            self.yaw_velocity = S

    def keyup(self, key):
        """ Update velocities based on key released
        Arguments:
            key: pygame key
        """
        if key == pygame.K_UP or key == pygame.K_DOWN:  # set zero forward/backward velocity
            self.for_back_velocity = 0
        elif key == pygame.K_LEFT or key == pygame.K_RIGHT:  # set zero left/right velocity
            self.left_right_velocity = 0
        elif key == pygame.K_w or key == pygame.K_s:  # set zero up/down velocity
            self.up_down_velocity = 0
        elif key == pygame.K_a or key == pygame.K_d:  # set zero yaw velocity
            self.yaw_velocity = 0
        elif key == pygame.K_t:  # takeoff
            self.tello.takeoff()
            # self.tello.move_up(30)
            self.yaw_velocity = 0
            self.up_down_velocity = 0
            self.for_back_velocity = 0
            self.send_rc_control = True
        elif key == pygame.K_l:  # land
            self.tello.land()
            self.send_rc_control = False

    def update(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(self.left_right_velocity, self.for_back_velocity, self.up_down_velocity,
                                       self.yaw_velocity)

    def reset_speed(self):
        """ Update routine. Send velocities to Tello."""
        if self.send_rc_control:
            self.tello.send_rc_control(0, 0, 0, 0)

def drone_thread():
    telloControl = TelloControl()
    telloControl.run()

threading.Thread(target=drone_thread, args=()).start()
