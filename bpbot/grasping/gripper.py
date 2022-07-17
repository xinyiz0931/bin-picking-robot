# -*- coding: utf-8 -*-
import os
import sys
import time
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from bpbot.utils import *

class Gripper(object):
    def __init__(self, finger_w, finger_h, open_w, gripper_size):
        self.open_w = open_w
        self.gripper_size = gripper_size
        self.finger_w = finger_w
        self.finger_h = finger_h
    
    # a series of transformations: sim <-> image <-> real world
    def image2real(self):
        ratio = 500/250
        self.open_w /= ratio
        self.gripper_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio
        
    def image2sim(self):
        ratio = 500/225
        self.open_w /= ratio
        self.gripper_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio

    def sim2image(self):
        ratio = 225/500
        self.open_w /= ratio
        self.gripper_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio    

    def sim2real(self):
        ratio = 225/250
        self.open_w /= ratio
        self.gripper_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio    

    def real2image(self):
        ratio = 250/500
        self.open_w /= ratio
        self.gripper_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio

    def real2sim(self):
        ratio = 250/225
        self.open_w /= ratio
        self.gripper_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio
    
    def print_gripper(self):
        notice_print(f"Finger=({self.finger_w},{self.finger_h}),open={self.open_w},size={self.gripper_size}")

    def create_hand_model(self):
        
        c = int(self.gripper_size/2)
        how = int(self.open_w/2)
        hfh = int(self.finger_h/2)
        fw = int(self.finger_w)

        ho = np.zeros((self.gripper_size, self.gripper_size), dtype = "uint8") # open
        hc = np.zeros((self.gripper_size, self.gripper_size), dtype = "uint8") # close
        
        ho[(c-hfh):(c+hfh), (c-how-fw):(c-how)]=255
        ho[(c-hfh):(c+hfh), (c+how):(c+how+fw)]=255
        hc[(c-hfh):(c+hfh), (c-how):(c+how)]=255

        return ho, hc

    def get_hand_model(self, model_type, model_width=None, model_height=None, open_width=None, x=None, y=None,radian=None):

        '''Note: angle is radian''' 
        if model_width is None:
            model_width = self.gripper_size
        if model_height is None:
            model_height = self.gripper_size
        if open_width is None:
            open_width = self.open_w
        how = int(open_width/2)
        hfh = int(self.finger_h/2)

        fw = int(self.finger_w)

        ho = np.zeros((model_width, model_height), dtype = "uint8") # open
        ho_line = np.zeros((model_width, model_height), dtype='uint8')
        hc = np.zeros((model_width, model_height), dtype = "uint8") # close

        if x is None and y is None:
            x, y = model_width/2, model_height/2
            
        x, y = int(x), int(y)

        ho[(y-hfh):(y+hfh), (x-how-fw):(x-how)]=255
        ho[(y-hfh):(y+hfh), (x+how):(x+how+fw)]=255
        
        ho_line[(y-hfh):(y+hfh), (x-how-fw):(x-how)]=255
        ho_line[(y-hfh):(y+hfh), (x+how):(x+how+fw)]=255
        ho_line[(y-1):(y+1), (x-how):(x+how)]=255

        hc[(y-hfh):(y+hfh), (x-how):(x+how)]=255
        if radian is None:
            if model_type=='open':
                return ho
            elif model_type == 'close':
                return hc
            elif model_type == 'shape':
                return ho_line
            else:
                return None

        else:
            angle = radian * 180/math.pi
            if model_type=='open':
                ho = Image.fromarray(np.uint8(ho))
                #angle = angle * 180/math.pi
                ho_ = ho.rotate(angle, center=(x,y))
                return np.array(ho_.convert('L'))
            
            elif model_type=='close':
                hc = Image.fromarray(np.uint8(hc))
                #angle = angle * 180/math.pi
                hc_ = hc.rotate(angle, center=(x,y))
                return np.array(hc_.convert('L'))
            elif model_type == 'shape':
                ho_line = Image.fromarray(np.int8(ho_line))
                ho_line_ = ho_line.rotate(angle, center=(x,y))
                return np.array(ho_line_.convert('L'))
            else:
                warning_print("Wrong input hand type")
                return None
        cv2.imshow("window", np.array(ho_.convert('L')))
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def draw_grasp(self, grasps, img, top_idx=0, color=(255,0,0),top_color=(0,255,0), top_only=False):
        # default: draw top grasp as No.0 of grasp list
        # e.g. top_no=3, draw top grasp as No.3
        grasps = np.array(grasps)
        if grasps.shape[0] < 1: return img

        if grasps.shape[1] > 3: # output from graspability
            grasps = grasps[:, [1, 2, 4]] 

        if top_only:
            x = int(grasps[top_idx][0])
            y = int(grasps[top_idx][1])
            angle = grasps[top_idx][2]
            # open_w = grasps[i][7]
            open_w = self.open_w
            h,w,_ = img.shape
            mask = self.get_hand_model('shape',h,w,open_w,x,y,angle)
            (r,g,b) = top_color
            rgbmask = np.ones((h, w), dtype="uint8")
            rgbmask = np.dstack((np.array(rgbmask * b, 'uint8'), np.array(rgbmask * g, 'uint8'),
                            np.array(rgbmask * r, 'uint8')))
            mask_resized = np.resize(mask, (h,w))
            img[:] = np.where(mask_resized[:h, :w, np.newaxis] == 0, img, rgbmask)
        else:
            for i in range(len(grasps)-1,-1,-1):
                x = int(grasps[i][0])
                y = int(grasps[i][1])
                angle = grasps[i][2]
                # open_w = grasps[i][7]
                open_w = self.open_w
                h,w,_ = img.shape
                mask = self.get_hand_model('shape',h,w,open_w,x,y,angle)
                if i == top_idx:
                    (r,g,b) = top_color
                else:
                    (r,g,b) = color
                rgbmask = np.ones((h, w), dtype="uint8")
                rgbmask = np.dstack((np.array(rgbmask * b, 'uint8'), np.array(rgbmask * g, 'uint8'),
                                np.array(rgbmask * r, 'uint8')))
                mask_resized = np.resize(mask, (h,w))
                img[:] = np.where(mask_resized[:h, :w, np.newaxis] == 0, img, rgbmask)
        return img
        
    def create_grasp_model(self, x, y, angle, width):
        left_x = int(x + (width/2)*math.cos(angle))
        left_y = int(y - (width/2)*math.sin(angle))
        right_x = int(x - (width/2)*math.cos(angle))
        right_y = int(y + (width/2)*math.sin(angle))
        return left_x, left_y, right_x, right_y

    def draw_grasp_ver2(self, grasps, img, top_color=(255,0,0)):
        """Use one line and an small empty circle representing grasp pose"""

        for i in range(len(grasps)-1,-1,-1):
            # [_,x,y,_,angle, ca, cb] = grasps[i]
            x = int(grasps[i][1])
            y = int(grasps[i][2])
            angle = grasps[i][4]
            ca = grasps[i][5]
            cb = grasps[i][6]
            ow = grasps[i][7]
            lx, ly, rx, ry = self.create_grasp_model(x,y,angle, ow) # on drawc
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

    def draw_single_grasp(self, x,y,angle, img, color):
        """Use one line and an small empty circle representing grasp pose"""
        lx, ly, rx, ry = self.grasp_model(x,y,angle) # on drawc
        (r,g,b) = color
        cv2.circle(img, (lx, ly), 7, (b,g,r), -1)
        cv2.circle(img, (rx, ry), 7, (b,g,r), -1)
        cv2.line(img, (lx, ly), (rx, ry), (b,g,r), 2)
        return img
    
    def point_oriented_grasp(self, img, loc):
        # for sr shape
        ow = 35
        down_depth = 25
        _p =25
        # for sc shape
        ow = 35
        # down_depth = 10
        # _p = 10 
        from bpbot.utils import adjust_grayscale, replace_bad_point
        start_rotation = 0
        stop_rotation = 180
        rotation_step = 10
        # (x,y) = replace_bad_point(img, loc)
        (x,y) = loc

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        if gray[x,y] < 5: (x,y) = replace_bad_point(img, loc)
        # gray = adjust_grayscale(gray, max_b=255)
        p_depth = gray[y,x]
        # print(p_depth)
        # print("pixel: ", p_depth) 
        
        # conflict with others mask
        # _, Wc = cv2.threshold(gray, max(1, p_depth-50), 255, cv2.THRESH_BINARY)
        _, Wc = cv2.threshold(gray, max(1, p_depth-down_depth), 255, cv2.THRESH_BINARY)
        
        # object touch mask
        Wt = gray.copy()
        Wt[Wt < (p_depth-_p)] = 0
        Wt[Wt > (p_depth)] = 0
        Wt[Wt > 0] = 255
        # _, Wt = cv2.threshold(gray, p_depth-3, 255, cv2.THRESH_BINARY)
        # print("max: ", Wt.max())

        # overlay = cv2.hconcat([Wc, Wt])
        # plt.imshow(overlay), plt.show() 
        
        max_r, max_g = 0, 0
        for r in np.arange(start_rotation, stop_rotation, rotation_step):
            radian = r * math.pi / 180
            Hc = self.get_hand_model('open',  open_width=ow, radian=radian)
            Ht = self.get_hand_model('close', open_width=ow, radian=radian)
            # overlay = cv2.hconcat([Hc, Ht])
            # plt.imshow(overlay) , plt.show()
            C = cv2.filter2D(Wc, -1, Hc) #Hc
            T = cv2.filter2D(Wt, -1, Ht) #Ht
            C_ = 255-C
            
            comb = T & C_
            G = cv2.GaussianBlur(comb, (75, 75), 25, 25)
            # vis1 = cv2.addWeighted(gray, 0.65, C_, 0.35, 1)
            # vis2 = cv2.addWeighted(gray, 0.65, T, 0.35, 1)
            # vis3= cv2.addWeighted(gray, 0.65, comb, 0.35, 1)
            # plt.imshow(cv2.hconcat([vis1, vis2, vis3])), plt.show()
            # print(r, " --> ", G[y,x])
            if G[y,x] >= max_g: 
                max_r = r
                max_g = G[y,x]
        return max_r