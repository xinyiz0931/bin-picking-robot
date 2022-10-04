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
    def __init__(self, finger_w, finger_h, open_w):
        self.open_w = open_w
        self.finger_w = finger_w
        self.finger_h = finger_h

        self.real2image()
        self.tplt_size = 500

    # a series of transformations: sim <-> image <-> real world
    def image2real(self):
        ratio = 500/250
        self.open_w /= ratio
        self.tplt_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio
        
    def image2sim(self):
        ratio = 500/225
        self.open_w /= ratio
        self.tplt_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio

    def sim2image(self):
        ratio = 225/500
        self.open_w /= ratio
        self.tplt_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio    

    def sim2real(self):
        ratio = 225/250
        self.open_w /= ratio
        self.tplt_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio    

    def real2image(self):
        ratio = 1/2
        self.open_w /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio

    def real2sim(self):
        ratio = 250/225
        self.open_w /= ratio
        self.tplt_size /= ratio
        self.finger_w /= ratio
        self.finger_h /= ratio
    
    def print_gripper(self):
        notice_print(f"Finger=({self.finger_w},{self.finger_h}),open={self.open_w},size={self.tplt_size}")

    def create_hand_model(self):
        
        c = int(self.tplt_size/2)
        how = int(self.open_w/2)
        hfh = int(self.finger_h/2)
        fw = int(self.finger_w)

        ho = np.zeros((self.tplt_size, self.tplt_size), dtype = "uint8") # open
        hc = np.zeros((self.tplt_size, self.tplt_size), dtype = "uint8") # close
        
        ho[(c-hfh):(c+hfh), (c-how-fw):(c-how)]=255
        ho[(c-hfh):(c+hfh), (c+how):(c+how+fw)]=255
        hc[(c-hfh):(c+hfh), (c-how):(c+how)]=255

        return ho, hc

    def get_hand_model(self, model_type, w=None, h=None, open_w=None, x=0, y=0,theta=0):
        """Open/closing model of gripper 

        Args:
            model_type (str): open/close/shape
            * when w,h,open_w all equals None, then use the values in the class
            * when x,y,theta(degree) all equals None, then use x=0,y=0,theta=0
        Returns:
            array: hand model image 
        """

        if w is None:
            w = self.tplt_size
        if h is None:
            h = self.tplt_size
        if open_w is None:
            open_w = self.open_w

        how = int(open_w/2)
        hfh = int(self.finger_h/2)

        fw = int(self.finger_w)

        
        ho = np.zeros((h, w), dtype = "uint8") # open
        ho_line = np.zeros((h, w), dtype='uint8')
        hc = np.zeros((h, w), dtype = "uint8") # close

        if x == 0 and y == 0:
            x, y = int(w/2), int(h/2)

        ho[(y-hfh):(y+hfh), (x-how-fw):(x-how)]=255
        ho[(y-hfh):(y+hfh), (x+how):(x+how+fw)]=255
        
        ho_line[(y-hfh):(y+hfh), (x-how-fw):(x-how)]=255
        ho_line[(y-hfh):(y+hfh), (x+how):(x+how+fw)]=255
        ho_line[(y-1):(y+1), (x-how):(x+how)]=255
        ho_line[(y-3):(y+3), (x-3):(x+3)]=255

        hc[(y-hfh):(y+hfh), (x-how):(x+how)]=255
        if theta == 0:
            if model_type=='open':
                return ho
            elif model_type == 'close':
                return hc
            elif model_type == 'shape':
                return ho_line
            else:
                return None
        else:
            if model_type=='open':
                ho = Image.fromarray(np.uint8(ho))
                ho_ = ho.rotate(theta, center=(x,y))
                return np.array(ho_.convert('L'))
            
            elif model_type=='close':
                hc = Image.fromarray(np.uint8(hc))
                hc_ = hc.rotate(theta, center=(x,y))
                return np.array(hc_.convert('L'))

            elif model_type == 'shape':
                ho_line = Image.fromarray(np.int8(ho_line))
                ho_line_ = ho_line.rotate(theta, center=(x,y))
                return np.array(ho_line_.convert('L'))
            else:
                warn_print("Wrong input hand type")
                return None
        cv2.imshow("window", np.array(ho_.convert('L')))
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def draw_grasp(self, grasps, img, top_idx=0, color=(255,0,0),top_color=(0,255,0), top_only=False):
        # default: draw top grasp as No.0 of grasp list
        # e.g. top_no=3, draw top grasp as No.3
        grasps = np.array(grasps)
        if len(grasps.shape)==1:
            grasps = np.asarray([grasps])
        
        if top_only:
            x = int(grasps[top_idx][0])
            y = int(grasps[top_idx][1])
            angle = grasps[top_idx][2]
            
            open_w = self.open_w
            h,w,_ = img.shape
            mask = self.get_hand_model('shape',w,h,open_w,x,y,angle)
            (r,g,b) = top_color
            rgbmask = np.ones((h, w), dtype="uint8")
            rgbmask = np.dstack((np.array(rgbmask * r, 'uint8'), np.array(rgbmask * g, 'uint8'),
                            np.array(rgbmask * b, 'uint8')))
            mask_resized = np.resize(mask, (h,w))
            img[:] = np.where(mask_resized[:h, :w, np.newaxis] == 0, img, rgbmask)
        else:
            for i in range(len(grasps)-1,-1,-1):
                x = int(grasps[i][0])
                y = int(grasps[i][1])
                angle = grasps[i][2]
                
                open_w = self.open_w
                h,w,_ = img.shape
                mask = self.get_hand_model('shape',w,h,open_w,x,y,angle)
                if i == top_idx:
                    (r,g,b) = top_color
                else:
                    (b,g,r) = color
                rgbmask = np.ones((h, w), dtype="uint8")
                rgbmask = np.dstack((np.array(rgbmask * r, 'uint8'), np.array(rgbmask * g, 'uint8'),
                                np.array(rgbmask * b, 'uint8')))
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
        """
        open_w (int): open width in pixel
        down_depth (int): pixels when gripper approaching object 
        """
        # for sr shape
        # ow = 35
        # down_depth = 25
        # _p =25
        # for sc shape
        open_w = 35
        # open_w = 20
        down_depth = 20
        _p = 5
        from bpbot.utils import adjust_grayscale, replace_bad_point

        start_rotation = 0
        stop_rotation = 180
        rotation_step = 22.5
        (x,y) = loc

        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape
        # print(f"[!] Old point: {loc} = {gray[y,x]}")
        # (x,y) = replace_bad_point(img, loc)
        if gray[y,x] <= 10: (x,y) = replace_bad_point(img, loc)
        # print(f"[*] New point: [{x}, {y}] = {gray[y,x]}")
        # test = img.copy()
        # cv2.circle(test, loc, 5, (0,0,255), 3)
        # cv2.circle(test, (x,y), 5, (0,255,0), -1)
        # cv2.imshow("", test), cv2.waitKey()

        # gray = adjust_grayscale(gray, max_b=255)
        p_depth = gray[y,x]
        # print(p_depth)
        # print("[*] pixel: ", p_depth) 
        
        # conflict with others mask
        # _, Wc = cv2.threshold(gray, max(1, p_depth-50), 255, cv2.THRESH_BINARY)
        # print(p_depth - down_depth)
        _, Wc = cv2.threshold(gray, max(1, p_depth-down_depth), 255, cv2.THRESH_BINARY)
        
        # object touch mask
        Wt = gray.copy()
        h,w = gray.shape

        Wt[Wt < (p_depth-_p)] = 0
        Wt[Wt > (p_depth)] = 0
        Wt[Wt > 0] = 255
        # Wt = cv2.resize(Wt, (500,500))
        # Wc = cv2.resize(Wc, (500,500))
        # Wt = cv2.resize(Wt, (500,500))
        
        # _, Wt = cv2.threshold(gray, p_depth-3, 255, cv2.THRESH_BINARY)
        # print("max: ", Wt.max())
        
        # overlay = cv2.hconcat([Wc, Wt])
        # plt.imshow(overlay), plt.title("Conflict mask & Touch mask")
        # plt.show() 
        
        max_r, max_g = 0, 0
        scores = []
        rotations = np.arange(start_rotation, stop_rotation, rotation_step)
        for r in rotations:
            Hc = self.get_hand_model('open', w=w,h=h,open_w=open_w,theta=r)
            Ht = self.get_hand_model('close',w=w,h=h,open_w=open_w,theta=r)
            rect = self.get_hand_model("close", w=w,h=h,open_w=open_w, x=x, y=y, theta=r)
            comb_Wt = Wt & rect

            # overlay = cv2.hconcat([Wc, comb_Wt])
            # plt.imshow(overlay), plt.title("Conflict mask & Touch mask")
            # plt.show()
            # overlay = cv2.hconcat([Hc, Ht])
            # plt.imshow(overlay) , plt.show()
            C = cv2.filter2D(Wc, -1, Hc) #Hc
            T = cv2.filter2D(comb_Wt, -1, Ht) #Ht
            C_ = 255-C
            
            comb = T & C_
            G = cv2.GaussianBlur(comb, (75, 75), 25, 25)
            # vis1 = cv2.addWeighted(gray, 0.65, C_, 0.35, 1)
            # vis2 = cv2.addWeighted(gray, 0.65, T, 0.35, 1)
            # vis3= cv2.addWeighted(gray, 0.65, comb, 0.35, 1)
            # plt.imshow(cv2.hconcat([vis1, vis2, vis3])), plt.show()
            scores.append(G[y,x])
            # print(r, " => ", G[y,x])
            if G[y,x] >= max_g: 
                max_r = r
                max_g = G[y,x]
        scores = np.asarray(scores)
        if np.count_nonzero(scores) == 0: 
            # print("Detection failed! ")
            return None
        else:
            return [x,y,rotations[np.argmax(scores)]]
        # print(scores) 
        # return (x,y,max_r)