# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import argparse
import configparser
import shutil
from datetime import datetime as dt
import timeit
sys.path.append("./")
import motion.motion_generator as mg
from utils.base_utils import warning_print
from utils.vision_utils import *

import cv2
import numpy as np
import matplotlib.pyplot as plt

class Gripper(object):
    def __init__(self, finger_w=15, finger_h=25, gripper_width=37, model_ratio=250/112.5):
        self.gripper_width = gripper_width
        self.model_ratio = model_ratio
        self.finger_w = finger_w
        self.finger_h = finger_h

    def hand_model(self, model_size=250):

        c = int(model_size/2)
        how = int(self.gripper_width/2*self.model_ratio)
        hft = int(self.finger_h/2*self.model_ratio)
        fw = int(self.finger_w*self.model_ratio)

        ho = np.zeros((model_size, model_size), dtype = "uint8") # open
        hc = np.zeros((model_size, model_size), dtype = "uint8") # close
        
        ho[(c-hft):(c+hft), (c-how-fw):(c-how)]=255
        ho[(c-hft):(c+hft), (c+how):(c+how+fw)]=255
        hc[(c-hft):(c+hft), (c-how):(c+how)]=255

        return ho, hc

    def grasp_model(self, x, y, angle):
        left_x = int(x + (self.gripper_width/3)*self.model_ratio*math.cos(angle))
        left_y = int(y - (self.gripper_width/3)*self.model_ratio*math.sin(angle))
        right_x = int(x - (self.gripper_width/3)*self.model_ratio*math.cos(angle))
        right_y = int(y + (self.gripper_width/3)*self.model_ratio*math.sin(angle))
        return left_x, left_y, right_x, right_y

    def draw_grasp(self, grasps, img, top_color=(255,0,0)):
        """Use one line and an small empty circle representing grasp pose"""
        for i in range(len(grasps)-1,-1,-1):
            # [_,x,y,_,angle, ca, cb] = grasps[i]
            x = int(grasps[i][1])
            y = int(grasps[i][2])
            angle = grasps[i][4]
            ca = grasps[i][5]
            cb = grasps[i][6]
            lx, ly, rx, ry = self.grasp_model(x,y,angle) # on drawc
            if i == 0:
                (r,g,b) = top_color
                cv2.circle(img, (lx, ly), 7, (b,g,r), -1)
                cv2.circle(img, (rx, ry), 7, (b,g,r), -1)
                cv2.line(img, (lx, ly), (rx, ry), (b,g,r), 2)
            else:
                color = 255-15*(i-1)
                cv2.circle(img, (lx, ly), 7, (0, color,color), -1)
                cv2.circle(img, (rx, ry), 7, (0, color,color), -1)
                cv2.line(img, (lx, ly), (rx, ry), (0, color,color), 2)
            # just for test
            # gmask = cv2.imread(f"./grasp/tmp/gmap_{ca}_{ca}.png",0)
            # plt.imshow(img, cmap='gray')
            # plt.imshow(gmask, alpha=0.5, cmap='jet')
            # plt.show()
        return img

    def draw_single_grasp(self, x,y,angle, img, color):
        """Use one line and an small empty circle representing grasp pose"""
        lx, ly, rx, ry = self.grasp_model(x,y,angle) # on drawc
        (r,g,b) = color
        cv2.circle(img, (lx, ly), 7, (b,g,r), -1)
        cv2.circle(img, (rx, ry), 7, (b,g,r), -1)
        cv2.line(img, (lx, ly), (rx, ry), (b,g,r), 2)
        return img

class Graspability(object):
    def __init__(self, rotation_step, depth_step, hand_depth):
        """
        rotation-step: iteration step for rotation [0,45,90,...]
        depth_step: iteration step for depth [0,30,60,..]
        hand_depth: distance of fingertip moving below grasp point before closing
        """
        self.rotation_step = rotation_step
        self.depth_step = depth_step
        self.hand_depth = hand_depth  # 50
    
    def rotate_img(self, img, angle, center=None, scale=1.0):
        (h,w) = img.shape[:2]

        if center is None:
            center=(w/2, h/2)

        M = cv2.getRotationMatrix2D(center, angle,scale)
        rotated = cv2.warpAffine(img, M, (w,h))
        return rotated
    
    def get_gaussian_blur(self, img, kernel_size=75, sigma = 25):
        return cv2.GaussianBlur(img,(kernel_size, kernel_size),sigma, sigma)
        
    def takefirst(self,elem):
        return elem[0]

    def graspability_map(self, img, hand_open_mask, hand_close_mask):
        """ generate graspability map and obtain all grasp candidates
        Parameters
        ----------
        img : 3-channel image
        hand_open_mask : 1-channel image
        hand_close_mask : 1-channel image

        Returns
        -------
        candidates : list  
            a list containing all possible grasp candidates, every candidate -> [g,x,y,z]
        """

        candidates = []
        count_a = 0
        count_b = 0
        start_rotation = 0
        stop_rotation = 180
        start_depth = 0
        stop_depth = 201
       
       
        # prepare rotated hand model
        ht_rot, hc_rot = [], []
        from PIL import Image
        hand_open_mask = Image.fromarray(np.uint8(hand_open_mask))
        hand_close_mask = Image.fromarray(np.uint8(hand_close_mask))

        for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
            ht_= hand_close_mask.rotate(r)
            hc_ = hand_open_mask.rotate(r)
            ht_rot.append(np.array(ht_.convert('L')))
            hc_rot.append(np.array(hc_.convert('L')))

        # print("Computing graspability map... ")
        # img = cv2.GaussianBlur(img,(3,3),0)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        for d in np.arange(start_depth, stop_depth, self.rotation_step):
            count_a += 1
            
            _, Wc = cv2.threshold(gray, d, 255, cv2.THRESH_BINARY)
            _, Wt = cv2.threshold(gray, d + self.hand_depth, 255, cv2.THRESH_BINARY)

            count_b = 0
            for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
                Hc = hc_rot[count_b]
                Ht = ht_rot[count_b]
                
                C = cv2.filter2D(Wc, -1, Hc) #Hc
                T = cv2.filter2D(Wt, -1, Ht) #Ht
                count_b += 1

                C_ = 255-C
                comb = T & C_
                G = self.get_gaussian_blur(comb)

                ret, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
                ccwt = cv2.connectedComponentsWithStats(thresh)
                
                res = np.delete(ccwt[3], 0, 0)

                for i in range(res[:,0].shape[0]):
                    y = int(res[:,1][i])
                    x = int(res[:,0][i])
                    z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                    angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                    candidates.append([G[y][x], x, y, z, angle, count_a, count_b])
        candidates.sort(key=self.takefirst, reverse=True)
        return candidates

    def target_oriented_graspability_map(self, img, hand_open_mask, hand_close_mask, Wc, Wt):
        """ generate graspability map and obtain all grasp candidates
        with known target object
        Parameters
        ----------
        img : 3-channel image
        hand_open_mask : 1-channel image
        hand_close_mask : 1-channel image

        Returns
        -------
        candidates : list  
            a list containing all possible grasp candidates, every candidate -> [g,x,y,z]
        """

        candidates = []
        count_a = 0
        count_b = 0
        start_rotation = 0
        stop_rotation = 180
        start_depth = 0
        stop_depth = 201
       
        # prepare rotated hand model
        ht_rot, hc_rot = [], []
        from PIL import Image
        hand_open_mask = Image.fromarray(np.uint8(hand_open_mask))
        hand_close_mask = Image.fromarray(np.uint8(hand_close_mask))

        for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
            ht_= hand_close_mask.rotate(r)
            hc_ = hand_open_mask.rotate(r)
            ht_rot.append(np.array(ht_.convert('L')))
            hc_rot.append(np.array(hc_.convert('L')))

        # print("Computing graspability map... ")
        # img = cv2.GaussianBlur(img,(3,3),0)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        

        count_b = 0
        for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
            Hc = hc_rot[count_b]
            Ht = ht_rot[count_b]
            
            C = cv2.filter2D(Wc, -1, Hc) #Hc
            T = cv2.filter2D(Wt, -1, Ht) #Ht
            count_b += 1

            C_ = 255-C
            comb = T & C_
            G = self.get_gaussian_blur(comb)

            ret, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
            ccwt = cv2.connectedComponentsWithStats(thresh)
            
            res = np.delete(ccwt[3], 0, 0)

            for i in range(res[:,0].shape[0]):
                y = int(res[:,1][i])
                x = int(res[:,0][i])
                z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                candidates.append([G[y][x], x, y, z, angle, count_a, count_b])

            # cv2.imwrite(f"./vision/tmp/G_{count_b}.png", G)
            # cv2.imwrite(f"./vision/tmp/C_{count_b}.png", C)
            # cv2.imwrite(f"./vision/tmp/CBar_{count_b}.png", C_)
            # cv2.imwrite(f"./vision/tmp/T_{count_b}.png", T)
            # cv2.imwrite(f"./vision/tmp/comb_{count_b}.png", comb)

        candidates.sort(key=self.takefirst, reverse=True)
        print(candidates[0])
        return candidates

    def combined_graspability_map(self, img, hand_open_mask, hand_close_mask, merge_mask):

        candidates = []
        count_a = 0
        count_b = 0
        start_rotation = 0
        stop_rotation = 180
        start_depth = 0
        stop_depth = 201
       
        # prepare rotated hand model
        ht_rot, hc_rot = [], []
        from PIL import Image
        hand_open_mask = Image.fromarray(np.uint8(hand_open_mask))
        hand_close_mask = Image.fromarray(np.uint8(hand_close_mask))

        for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
            ht_= hand_close_mask.rotate(r)
            hc_ = hand_open_mask.rotate(r)
            ht_rot.append(np.array(ht_.convert('L')))
            hc_rot.append(np.array(hc_.convert('L')))

        # print("Computing graspability map... ")
        # img = cv2.GaussianBlur(img,(3,3),0)
        merge_mask = 255 - cv2.resize(merge_mask, (img.shape[1], img.shape[0]))


        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for d in np.arange(start_depth, stop_depth, self.rotation_step):
            count_a += 1
            
            _, Wc = cv2.threshold(gray, d, 255, cv2.THRESH_BINARY)
            _, _Wt = cv2.threshold(gray, d + self.hand_depth, 255, cv2.THRESH_BINARY)
            Wt = cv2.bitwise_and(merge_mask,_Wt)
            
            # plt.imshow(Wt)
            # plt.show()


            count_b = 0
            for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
                Hc = hc_rot[count_b]
                Ht = ht_rot[count_b]

                HtMask = np.array(Ht/255000, dtype="float32")
                T = cv2.filter2D(Wt, -1, HtMask) 
                # T = cv2.filter2D(Wt, -1, Ht)
                C = cv2.filter2D(Wc, -1, Hc) 

                count_b += 1

                C_ = 255-C
                comb = T & C_
                G = self.get_gaussian_blur(comb)
                

                ret, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
                ccwt = cv2.connectedComponentsWithStats(thresh)
                
                res = np.delete(ccwt[3], 0, 0)

                for i in range(res[:,0].shape[0]):
                    y = int(res[:,1][i])
                    x = int(res[:,0][i])
                    z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                    angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                    candidates.append([G[y][x], x, y, z, angle, count_a, count_b])
                # save all images during the process                
        return candidates

    def grasp_detection(self, candidates, w, h, n=10, _dismiss=50, _distance=50):
        """Detect grasp with graspability

        Arguments:
            candidates {list} -- a list of grasps candidates
            w {int} -- width
            h {int} -- height

        Keyword Arguments:
            n {int} -- number of grasp (default: {20})
            _dismiss {int} -- distance that will not collide with the box (default: {25})

        Returns:
            [list] -- a list of executable including [g,x,y,z,a,rot_step, depth_step]
        """

        # candidates.sort(key=self.takefirst, reverse=True)
        i = 0
        k = 0
        grasps = []
        # print("Computing grasps! ")
        if len(candidates) < n:
            n = len(candidates)
        # while (len(candidates) and i <= n):
        while (len(candidates) and i < n):
            x = candidates[i][1]
            y = candidates[i][2]
            ## consider dismiss/distance to rank grasp candidates
            if (_dismiss < x) and (x < w-_dismiss) and (_dismiss < y) and (y < h-_dismiss):
            # if (x < w-_dismiss) and (y < h-_dismiss):
                if grasps == []:
                    grasps.append(candidates[i])
                    k += 1
                else:
                    # check the distance of this candidate and all others 
                    g_array = np.array(grasps)
                    x_array = (np.ones(len(grasps)))*x
                    y_array = (np.ones(len(grasps)))*y
                    _d_array = (np.ones(len(grasps)))*_distance
                    if ((x_array - g_array[:,1])**2+(y_array - g_array[:,2])**2 > _d_array**2).all():
                        grasps.append(candidates[i])
                        k += 1
            i += 1
            if k > n:
                break
        return grasps

