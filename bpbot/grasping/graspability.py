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
from cv2 import merge

#import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bpbot.utils import *

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
        """Generate graspability map

        Args:
            img (array): W x H x 3
            hand_open_mask (array): 500 x 500 x 1
            hand_close_mask (arrau): 500 x 500 x1

        Returns:
            grasps (array): N * [g_score, x, y, r(degree)]
        """

        candidates = []
        start_rotation = 0
        stop_rotation = 180
        start_depth = 0
        stop_depth = 201
       
       
        # prepare rotated hand model
        ht_rot, hc_rot = [], []
        
        hand_open_mask = Image.fromarray(np.uint8(hand_open_mask))
        hand_close_mask = Image.fromarray(np.uint8(hand_close_mask))

        for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
            
            ht_= hand_close_mask.rotate(r)
            hc_ = hand_open_mask.rotate(r)
            ht_rot.append(np.array(ht_.convert('L')))
            hc_rot.append(np.array(hc_.convert('L'))) 
            # print("xinyi | ", r)
            # plt.imshow(ht_.convert('L')), plt.show()
        # print("Computing graspability map... ")
        # img = cv2.GaussianBlur(img,(3,3),0)

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        
        for d_idx, d in enumerate(np.arange(start_depth, stop_depth, self.rotation_step)):
            # count_a += 1
            
            _, Wc = cv2.threshold(gray, d, 255, cv2.THRESH_BINARY)
            _, Wt = cv2.threshold(gray, d + self.hand_depth, 255, cv2.THRESH_BINARY)

            # count_b = 0
            for r_idx, r in enumerate(np.arange(start_rotation, stop_rotation, self.rotation_step)):
                Hc = hc_rot[r_idx]
                Ht = ht_rot[r_idx]
                # plt.imshow(cv2.hconcat([Hc, Ht])), plt.show()
                
                C = cv2.filter2D(Wc, -1, Hc) #Hc
                T = cv2.filter2D(Wt, -1, Ht) #Ht
                # count_b += 1

                C_ = 255-C
                comb = T & C_
                G = self.get_gaussian_blur(comb)

                ret, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
                ccwt = cv2.connectedComponentsWithStats(thresh)
                
                res = np.delete(ccwt[3], 0, 0)

                for i in range(res[:,0].shape[0]):
                    y = int(res[:,1][i])
                    x = int(res[:,0][i])
                    # z = int(self.depth_step*(d_idx-1)+self.hand_depth/2)
                    # angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                    #candidates.append([G[y][x], x, y, z, angle, count_a, count_b])
                    candidates.append([G[y][x],x,y,r])
        candidates.sort(key=self.takefirst, reverse=True)
        return candidates
    
    def width_adjusted_graspability_map(self, img, hand_open_mask, hand_close_mask, width_count):
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
        
        for d in np.arange(start_depth, stop_depth, self.depth_step):
            count_a += 1
            
            # revised by xinyi
            _, Wc = cv2.threshold(gray, d + self.hand_depth, 255, cv2.THRESH_BINARY)
            _, Wt = cv2.threshold(gray, d, 255, cv2.THRESH_BINARY)

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

                """Sample points with highest graspability"""
                _,maxG,_,maxLoc = cv2.minMaxLoc(G)
                x = maxLoc[0]
                y = maxLoc[1]
                z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                candidates.append([maxG, x, y, z, angle, count_a, count_b, width_count])
                
                """Sample points with largest graspability area"""
                # ret, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
                # ccwt = cv2.connectedComponentsWithStats(thresh)
                # res = np.delete(ccwt[3], 0, 0)
                # if res[:,0].shape[0]:
                #     y = int(res[:,1][0])
                #     x = int(res[:,0][0])
                #     z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                #     angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                #     candidates.append([G[y][x], x, y, z, angle, count_a, count_b, width_count])
                    
                """Sample points with all graspability area"""
                # ret, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
                # ccwt = cv2.connectedComponentsWithStats(thresh)
                
                # res = np.delete(ccwt[3], 0, 0)

                # for i in range(res[:,0].shape[0]):
                #     y = int(res[:,1][i])
                #     x = int(res[:,0][i])
                #     z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                #     angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                #     candidates.append([G[y][x], x, y, z, angle, count_a, count_b, width_count])

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

            _,maxG,_,maxLoc = cv2.minMaxLoc(G)
            x = maxLoc[0]
            y = maxLoc[1]
            z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
            angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
            candidates.append([maxG, x, y, z, angle, count_a, count_b])
            # for i in range(res[:,0].shape[0]):
            #     y = int(res[:,1][i])
            #     x = int(res[:,0][i])
            #     z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
            #     angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
            #     candidates.append([G[y][x], x, y, z, angle, count_a, count_b])

            # cv2.imwrite(f"./vision/tmp/G_{count_b}.png", G)
            # cv2.imwrite(f"./vision/tmp/C_{count_b}.png", C)
            # cv2.imwrite(f"./vision/tmp/CBar_{count_b}.png", C_)
            # cv2.imwrite(f"./vision/tmp/T_{count_b}.png", T)
            # cv2.imwrite(f"./vision/tmp/comb_{count_b}.png", comb)

        candidates.sort(key=self.takefirst, reverse=True)
        return candidates

    def point_oriented_graspability_map(self, img, loc, hand_open_mask, hand_close_mask, Wc, Wt):
        candidates = []
        count_b = 0
        start_rotation = 0
        stop_rotation = 180

        ht_rot, hc_rot = [], []
        from PIL import Image
        hand_open_mask = Image.fromarray(np.uint8(hand_open_mask))
        hand_close_mask = Image.fromarray(np.uint8(hand_close_mask))

        for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
            ht_ = hand_close_mask.rotate(r)
            hc_ = hand_open_mask.rotate(r)
            ht_rot.append(np.array(ht_.convert('L')))
            hc_rot.append(np.array(hc_.convert('L')))

        (x,y) = loc
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        p_depth = gray[y,x]
        
        _, Wc = cv2.threshold(gray, p_depth-self.hand_dept, 255, cv2.THRESH_BINARY)
        _, Wt = cv2.threshold(gray, p_depth, 255, cv2.THRESH_BINARY)
        
        max_r, max_g = 0, 0

        for r in np.arange(start_rotation, stop_rotation, self.rotation_step):
            Hc = hc_rot[count_b]
            Ht = ht_rot[count_b]
            C = cv2.filter2D(Wc, -1, Hc) #Hc
            T = cv2.filter2D(Wt, -1, Ht) #Ht
            count_b += 1

            C_ = 255-C
            comb = T & C_
            G = self.get_gaussian_blur(comb)

            if G.max() >= max_g: 
                max_r = r
                max_g = G.max()

        return max_r*(math.pi)/180 
 
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
        merge_mask /= (merge_mask.max() / 255.0) # real writhe value
        merge_mask = np.uint8(255 - cv2.resize(merge_mask, (img.shape[1], img.shape[0])))


        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        for d in np.arange(start_depth, stop_depth, self.rotation_step):
            count_a += 1
            
            _, Wc = cv2.threshold(gray, d, 255, cv2.THRESH_BINARY)
            _, _Wt = cv2.threshold(gray, d + self.hand_depth, 255, cv2.THRESH_BINARY)

            Wt = cv2.bitwise_and(merge_mask,_Wt)
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

                """Sample points with highest graspability"""
                _,maxG,_,maxLoc = cv2.minMaxLoc(G)
                x = maxLoc[0]
                y = maxLoc[1]
                z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                candidates.append([maxG, x, y, z, angle, count_a, count_b])
                
                """Sample points with largest graspability area"""
                # ret, thresh = cv2.threshold(comb, 122,255, cv2.THRESH_BINARY)
                # ccwt = cv2.connectedComponentsWithStats(thresh)
                # res = np.delete(ccwt[3], 0, 0)
                # for i in range(res[:,0].shape[0]):
                #     y = int(res[:,1][i])
                #     x = int(res[:,0][i])
                #     z = int(self.depth_step*(count_a-1)+self.hand_depth/2)
                #     angle = (start_rotation+self.rotation_step*(count_b-1))*(math.pi)/180
                #     candidates.append([G[y][x], x, y, z, angle, count_a, count_b])

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
            [array] -- an array of sorted grasps [[x,y,r (degree)], ...]
        """

        candidates.sort(key=self.takefirst, reverse=True)
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
        return np.asarray(grasps)[:,1:]

