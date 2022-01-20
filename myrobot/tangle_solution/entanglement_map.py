import os
import sys
import glob
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import numpy as np
import itertools

np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from scipy import ndimage
from utils.base_utils import *
from utils.image_proc_utils import *
from utils.plot_utils import *
from tangle_solution.topo_coor import TopoCoor, LineDetection

class EntanglementMap(object):
    def __init__(self, length_thre, distance_thre, sliding_size, sliding_stride):
        """Initialize EntanglementMap class with following four parameters

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
        self.tc = TopoCoor(length_thre, distance_thre)

    def entanglement_map(self, src):
        """
        Input: 3-channel image
        Output: entanglement map as an array, not an image in range[0,255]
        """

        height, width, _ = src.shape

        wmap = np.array([])
        dmap = np.array([])

        for y in range(0,height-self.sliding_size + 1, self.sliding_stride): 
            for x in range(0,width-self.sliding_size + 1, self.sliding_stride): 
                cropped = src[y:y + self.sliding_size , x:x + self.sliding_size]
                wmat, w, d = self.tc.topo_coor_from_image(cropped,cmask=False)
                wmap = np.append(wmap, w)
                dmap = np.append(dmap, d)

        hnum = int((height-self.sliding_size) / self.sliding_stride + 1)
        wnum = int((width-self.sliding_size) / self.sliding_stride + 1)

        wmap = (wmap.reshape(hnum, wnum)).astype(float)
        dmap = (dmap.reshape(hnum, wnum)).astype(float)
        
        wmat, w, d, cmask = self.tc.topo_coor_from_image(src,cmask=True)
        result_print("w: {:.2}, d: {:.2}".format(w,d))
        cmask = cv2.resize(cmask,(hnum, wnum))
        wmat /= (wmat.max() / 255.0) # real writhe value
        wmat_vis = np.uint8(wmat) # normalize to [0,255]

        emap = 0.8*np.nan_to_num(wmap)+0.2*np.nan_to_num(dmap)+0.001*cmask

        return emap, wmat_vis, w, d
    
    def line_map(self, src):
        """
        Input: 3-channel image
        Output: line number map as an array, not an image in range[0,255]
        """
        height, width, _ = src.shape

        lmap = np.array([])
        for y in range(0,height-self.sliding_size + 1, self.sliding_stride): 
            for x in range(0,width-self.sliding_size + 1, self.sliding_stride): 
                cropped = src[y:y + self.sliding_size , x:x + self.sliding_size]
                ld = LineDetection()
                _, _, lines_num = ld.detect_line(cropped,self.length_thre, self.distance_thre)
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
        Input: 3-channel image
        Output: brightness map as an array, not an image in range[0,255]
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

    