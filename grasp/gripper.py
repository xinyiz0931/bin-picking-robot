# -*- coding: utf-8 -*-
import os
import sys
import math

sys.path.append("./")
from utils.base_utils import *
from utils.image_proc_utils import rotate_img

import cv2
import numpy as np
import matplotlib.pyplot as plt


class Gripper(object):
    def __init__(self, handwidth=34, model_ratio=250 / 112.5):
        self.handwidth = handwidth
        self.model_ratio = model_ratio

    def hand_model(self, finger_w, finger_h, model_size=250):

        c = int(model_size / 2)
        how = int(self.handwidth / 2 * self.model_ratio)
        hft = int(finger_h / 2 * self.model_ratio)
        fw = int(finger_w * self.model_ratio)

        ho = np.zeros((model_size, model_size), dtype="uint8")  # open
        hc = np.zeros((model_size, model_size), dtype="uint8")  # close

        ho[(c - hft):(c + hft), (c - how - fw):(c - how)] = 255
        ho[(c - hft):(c + hft), (c + how):(c + how + fw)] = 255
        hc[(c - hft):(c + hft), (c - how):(c + how)] = 255

        return ho, hc

    def draw_model(self, finger_w, finger_h, close_width, model_size):

        c = int(model_size / 2)
        how = int(close_width / 2 * self.model_ratio)
        hft = int(finger_h / 2 * self.model_ratio)
        fw = int(finger_w * self.model_ratio)

        hand = np.zeros((model_size, model_size), dtype="uint8")  # open

        hand[(c - hft):(c + hft), (c - how - fw):(c - how)] = 255
        hand[(c - hft):(c + hft), (c + how):(c + how + fw)] = 255
        hand[(c - 1):(c + 1), (c - how):(c + how)] = 255
        return hand

    def grasp_model(self, x, y, angle):
        left_x = int(x + (self.handwidth / 2) * self.model_ratio * math.cos(angle))
        left_y = int(y - (self.handwidth / 2) * self.model_ratio * math.sin(angle))
        right_x = int(x - (self.handwidth / 2) * self.model_ratio * math.cos(angle))
        right_y = int(y + (self.handwidth / 2) * self.model_ratio * math.sin(angle))
        return left_x, left_y, right_x, right_y

    def draw_grasp(self, grasp, drawf, drawc, lm, tm):
        """Draw grasp on both cropped and full images

        Arguments:
            grasp {list} -- [graspability_score, x,y,z,angle]
            drawf {image} -- full image
            drawc {image} -- cropped image
            lm {int} -- left margin
            tm {int} -- top margin

        Returns:
            Drawn images
        """
        [_, x, y, _, angle] = grasp
        # fully information
        xf, yf, anglef = x + lm, y + tm, angle

        lx, ly, rx, ry = self.grasp_model(x, y, angle)  # on drawc
        lxf, lyf, rxf, ryf = self.grasp_model(xf, yf, anglef)  # on drawf

        cv2.circle(drawc, (lx, ly), 7, (0, 0, 255), -1)
        cv2.circle(drawc, (rx, ry), 7, (0, 0, 255), -1)
        cv2.circle(drawc, (x, y), 5, (0, 0, 255), -1)
        cv2.line(drawc, (lx, ly), (rx, ry), (0, 0, 255), 4)
        # draw full
        cv2.circle(drawf, (lxf, lyf), 7, (0, 0, 255), -1)
        cv2.circle(drawf, (rxf, ryf), 7, (0, 0, 255), -1)
        cv2.circle(drawf, (xf, yf), 5, (0, 0, 255), -1)
        cv2.line(drawf, (lxf, lyf), (rxf, ryf), (0, 0, 255), 4)
        return drawc, drawf

    def draw_uniform_grasps(self, grasps, drawf, drawc, lm, tm):
        """Draw grasp on a single image

        Arguments:
            grasps {list} -- [grasp1, grasp2,...], grasp1 = [g,x,y,z,a]
            draw {image} -- input image

        Returns:
            drawn image
        """
        for i in range(len(grasps)):
            # cropped information
            [_, x, y, _, angle] = grasps[i]
            # fully information
            xf, yf, anglef = x + lm, y + tm, angle

            lx, ly, rx, ry = self.grasp_model(x, y, angle)  # on drawc
            lxf, lyf, rxf, ryf = self.grasp_model(xf, yf, anglef)  # on drawf
            # draw cropped
            cv2.circle(drawc, (lx, ly), 7, (0, 255, 255), -1)
            cv2.circle(drawc, (rx, ry), 7, (0, 255, 255), -1)
            cv2.circle(drawc, (x, y), 5, (0, 255, 255), -1)
            cv2.line(drawc, (lx, ly), (rx, ry), (0, 255, 255), 4)
            # draw full
            cv2.circle(drawf, (lxf, lyf), 7, (0, 255, 255), -1)
            cv2.circle(drawf, (rxf, ryf), 7, (0, 255, 255), -1)
            cv2.circle(drawf, (xf, yf), 5, (0, 255, 255), -1)
            cv2.line(drawf, (lxf, lyf), (rxf, ryf), (0, 255, 255), 4)

        cv2.imwrite("./vision/depth/candidate_f.png", drawf)
        cv2.imwrite("./vision/depth/candidate_c.png", drawc)
        return drawc, drawf

    def draw_grasp_single_image(self, grasps, img, r, g, b):
        mask_size = 120
        h, w, _ = img.shape
        bgmask = np.zeros((h, w), dtype="uint8")
        bggrasp = np.ones((h, w), dtype="uint8")
        bggrasp = np.dstack((np.array(bggrasp * b, 'uint8'), np.array(bggrasp * g, 'uint8'),
                             np.array(bggrasp * r, 'uint8')))

        for i in range(len(grasps)):
            [_, x, y, _, angle] = grasps[i]
            hand_mask = self.draw_model(finger_w=5, finger_h=13, close_width=28, model_size=120)
            mask = rotate_img(hand_mask, angle)

            offset = mask_size / 2
            bgmask[int(y - offset):int(y + offset), int(x - offset):int(x + offset)] = mask
            # use the color you want in rgb channel
        img[:] = np.where(bgmask[:h, :w, np.newaxis] == 0, img, bggrasp)
        # draw point
        for i in range(len(grasps)):
            [_, x, y, _, angle] = grasps[i]
            cv2.circle(img, (x, y), 5, (b, g, r), -1)

        return img

    def draw_grasps(self, grasps, drawf, drawc, lm, tm, all=True):
        """Use one line and an small empty circle representing grasp pose"""
        for i in range(len(grasps)):
            # cropped information
            [_, x, y, _, angle] = grasps[i]
            # fully information
            xf, yf, anglef = x + lm, y + tm, angle
            if all == True:
                lx, ly, rx, ry = self.grasp_model(x, y, angle)  # on drawc
                lxf, lyf, rxf, ryf = self.grasp_model(xf, yf, anglef)  # on drawf
                if i == 0:
                    # draw cropped
                    cv2.circle(drawc, (lx, ly), 7, (0, 0, 255), -1)
                    cv2.circle(drawc, (rx, ry), 7, (0, 0, 255), -1)
                    cv2.circle(drawc, (x, y), 5, (0, 0, 255), -1)
                    cv2.line(drawc, (lx, ly), (rx, ry), (0, 0, 255), 4)
                    # draw full
                    cv2.circle(drawf, (lxf, lyf), 7, (0, 0, 255), -1)
                    cv2.circle(drawf, (rxf, ryf), 7, (0, 0, 255), -1)
                    cv2.circle(drawf, (xf, yf), 5, (0, 0, 255), -1)
                    cv2.line(drawf, (lxf, lyf), (rxf, ryf), (0, 0, 255), 4)
                else:
                    color = 255 - 15 * (i - 1)
                    # draw cropped
                    cv2.circle(drawc, (lx, ly), 7, (0, color, color), -1)
                    cv2.circle(drawc, (rx, ry), 7, (0, color, color), -1)
                    cv2.circle(drawc, (x, y), 5, (0, color, color), -1)
                    cv2.line(drawc, (lx, ly), (rx, ry), (0, color, color), 2)
                    # draw full
                    cv2.circle(drawf, (lxf, lyf), 7, (0, color, color), -1)
                    cv2.circle(drawf, (rxf, ryf), 7, (0, color, color), -1)
                    cv2.circle(drawf, (xf, yf), 5, (0, color, color), -1)
                    cv2.line(drawf, (lxf, lyf), (rxf, ryf), (0, color, color), 2)
            else:
                lx, ly, rx, ry = self.grasp_model(x, y, angle)  # on drawc
                lxf, lyf, rxf, ryf = self.grasp_model(xf, yf, anglef)  # on drawf
                # draw cropped
                cv2.circle(drawc, (lx, ly), 7, (0, 0, 255), -1)
                cv2.circle(drawc, (rx, ry), 7, (0, 0, 255), -1)
                cv2.circle(drawc, (x, y), 5, (0, 0, 255), -1)
                cv2.line(drawc, (lx, ly), (rx, ry), (0, 0, 255), 4)
                # draw full
                cv2.circle(drawf, (lx, ly), 7, (0, 0, 255), -1)
                cv2.circle(drawf, (rx, ry), 7, (0, 0, 255), -1)
                cv2.circle(drawf, (x, y), 5, (0, 0, 255), -1)
                cv2.line(drawf, (lx, ly), (rx, ry), (0, 0, 255), 4)
                break
        cv2.imwrite("./vision/depth/grasp_f.png", drawf)
        cv2.imwrite("./vision/depth/grasp_c.png", drawc)
        # from utils.image_proc_utils import rotate_img
        # cv2.imwrite("./testmyview.png", rotate_img(drawc,180))
        return drawc, drawf


if __name__ == '__main__':
    # img = cv2.imread("../vision/depth/final_result.png")
    # grasps = [[6, 500, 392, 2, -45], [1, 300, 450, 3, 150]]
    # mask_size = 120
    # gripper = Gripper(handwidth=28)
    # draw = gripper.draw_grasp_single_image(grasps, img, 255, 255, 0)
    # draw = gripper.draw_grasp_single_image([grasps[0]], img, 255, 0, 0)

    # cv2.imshow("windows", draw)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    gray = cv2.imread("D:\\code\\myrobot\\vision\\test\\test.png", 0)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    dot = np.multiply(gray, binary)
    plt.imshow(gray)
    plt.show()
    plt.imshow(dot)
    plt.show()




    # gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # h,w,_ = img.shape
    #
    # bgmask = np.zeros((h, w), dtype="uint8")  # open
    # bggrasp = np.ones((h, w), dtype="uint8")  # open
    #
    # offset = mask_size/2
    # bgmask[int(grasp[0]-offset):int(grasp[0]+offset), int(grasp[1]-offset):int(grasp[1]+offset)] = mask
    #
    # # use the color you want in rgb channel
    # (r,g,b) = (0,162,223)
    #
    # bggrasp = np.dstack((np.array(bggrasp * 255, 'uint8'), np.array(bggrasp * g, 'uint8'),
    #                         np.array(bggrasp * r, 'uint8')))
    # img[:] = np.where(bgmask[:h, :w, np.newaxis] == 0, img, bggrasp)
    #
    #
    # cv2.imshow("windows", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
