import os
import sys
import glob
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from bpbot.utils import *
from bpbot.tangle_solution import TopoCoor, LineDetection

class EntanglementMap(object):
    def __init__(self, length_thre, distance_thre, sliding_size, sliding_stride, 
                 weight_w=0.8, weight_d=0.199, weight_c=0.001):
        """
        Initialize EntanglementMap class with following four parameters
        Arguments:
            length_thre {int} -- length threshold when fitting line segments
            distance_thre {int} -- distance threshold when fitting line segments
            sliding_size {int} -- sliding window size !Must larger than object
            sliding_stride {int} -- sliding window stride
        """
        self.length_thre = length_thre
        self.distance_thre = distance_thre
        self.sliding_size = sliding_size
        self.sliding_stride = sliding_stride
        self.weight_w = weight_w
        self.weight_d = weight_d
        self.weight_c = weight_c

        self.ld = LineDetection(self.length_thre, self.distance_thre)
        self.tc = TopoCoor()

    def entanglement_map(self, src):
        """
        Calculate entanglement map for input depth image
        Parameters: 
            src {array} -- 3-channel image
        Return: 
            emap {array} -- (win num x win num), entanglement map using sliding window function (not normalized)
            wmat_vis {array} -- writhe matrix normalized in [0,255]
            w {float} -- writhe
            d {float} -- density
        """

        height, width, _ = src.shape

        wmap = np.array([])
        dmap = np.array([])

        for y in range(0,height-self.sliding_size + 1, self.sliding_stride): 
            for x in range(0,width-self.sliding_size + 1, self.sliding_stride): 
                cropped = src[y:y + self.sliding_size , x:x + self.sliding_size]
                wmat, w, d = self.tc.topo_coor_from_img(cropped,self.length_thre,self.distance_thre,cmask=False)
                wmap = np.append(wmap, w)
                dmap = np.append(dmap, d)

        hnum = int((height-self.sliding_size) / self.sliding_stride + 1)
        wnum = int((width-self.sliding_size) / self.sliding_stride + 1)
        wmap = (wmap.reshape(hnum, wnum)).astype(float)
        dmap = (dmap.reshape(hnum, wnum)).astype(float)
        
        # calculate weights
        wmat, w, d, cmask = self.tc.topo_coor_from_img(src,self.length_thre,self.distance_thre,cmask=True)
        result_print("w: {:.2}, d: {:.2}".format(w,d))
        cmask = cv2.resize(cmask,(hnum, wnum))
        wmat /= (wmat.max() / 255.0) # real writhe value
        wmat_vis = np.uint8(wmat) # normalize to [0,255]

        emap = self.weight_w*np.nan_to_num(wmap)+self.weight_d*np.nan_to_num(dmap)+self.weight_c*cmask

        return emap, wmat_vis, w, d
    
    def line_map(self, src):
        """
        Calculate a map for the distribution of line number from depth image
        Parameters: 
            src {array} -- 3-channel image
        Return: 
            lmap {array} -- line number map using sliding window function
        """
        height, width, _ = src.shape

        lmap = np.array([])
        for y in range(0,height-self.sliding_size + 1, self.sliding_stride): 
            for x in range(0,width-self.sliding_size + 1, self.sliding_stride): 
                cropped = src[y:y + self.sliding_size , x:x + self.sliding_size]
                _, _, lines_num = self.ld.detect_line(cropped)
                if lines_num is None:
                    lmap = np.append(lmap, 0)
                else:
                    lmap = np.append(lmap, lines_num)

        hnum = int((height-self.sliding_size) / self.sliding_stride + 1)
        wnum = int((width-self.sliding_size) / self.sliding_stride + 1)

        lmap = lmap.reshape(hnum, wnum)
        return lmap

    def brightness_map(self, src, brightness_thre=127):
        """
        Calculate a map for the distribution of pixel brightness from depth image
        Parameters: 
            src {array} -- 3-channel image
        Return: 
            bmap {array} -- pixel brightness map using sliding window function
        """
        height, width, _ = src.shape

        bmap = np.array([])

        for y in range(0,height-self.sliding_size + 1, self.sliding_stride): 
            for x in range(0,width-self.sliding_size + 1, self.sliding_stride): 
                cropped = src[y:y + self.sliding_size , x:x + self.sliding_size]
                cube = np.where(cropped >= brightness_thre)
                bmap = np.append(bmap, len(cube[0])/(height*width))

        hnum = int((height-self.sliding_size) / self.sliding_stride + 1)
        wnum = int((width-self.sliding_size) / self.sliding_stride + 1)
        bmap = bmap.reshape(hnum, wnum)
        return bmap

    