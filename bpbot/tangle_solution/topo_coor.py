"""
A python scripts to generate toopology coordinate (WM;w,d,c)
Arthor: Xinyi
Date: 2021/10/15
"""
import os
import sys
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import numpy as np
import matplotlib.pyplot as plt
from bpbot.utils import *

class LineDetection(object):
    def __init__(self, len_thld=15, dist_thld=3):
        # FLD instance, important in accuracy
        self.len_thld = len_thld
        self.dist_thld = dist_thld
        self.canny_aperture_size = 3
        self.canny_thre1= 50
        self.canny_thre2 = 50

    def detect_line(self, src, len_thld=None, dist_thld=None, vis=False):
        """
        FLD detector with depth information from depth images
        * If source image has no lines, return all None

        Parameters:
            src {array}           -- images with 3 channels
        Return: 
            lines_2d {array}      -- (num, 1, 4), without depth
            lines_reshape {array} -- (num, 1, 6), with depth info
            num {int}             -- number of detected edge segments
            *drawn {array}        -- image same size as src, if vis=True
                                  -- if vis=False, not return drawn
        """
        gray = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
        if len_thld is None: len_thld = self.len_thld
        if dist_thld is None: dist_thld = self.dist_thld

        fld = cv2.ximgproc.createFastLineDetector(len_thld,
                                                  dist_thld,
                                                  self.canny_thre1,
                                                  self.canny_thre2,
                                                  self.canny_aperture_size)
                                                #   do_merge=False)
        lines_2d = fld.detect(gray)
        if lines_2d is None:
            if vis:
                return src, None
            else:
                return None, None, None
        else:
            num = lines_2d.shape[0]
            lines_reshape = lines_2d.reshape(num, 4)
            b_start = []
            b_end = []
            for each in lines_2d:
                x1, y1, x2, y2 = np.int0(each[0])
                b1 = get_max_neighbor_pixel(gray, (x1, y1))
                b2 = get_max_neighbor_pixel(gray, (x2, y2))
                b_start = np.append(b_start, b1)
                b_end = np.append(b_end, b2)
            lines_reshape = np.insert(lines_reshape, 2, b_start, axis=1)
            lines_reshape = np.insert(lines_reshape, 5, b_end, axis=1)

            if vis is True:

                src_for_draw = src.copy()
                drawn = self.draw_line_segment(src_for_draw, lines_2d, thickness=2)

                # <-------------------- revised for colored lines ---------------------->
                # src_for_draw = src.copy()
                # cmap = get_cmap(num)
                # # for i in range(14):
                # #     print(rgba2rgb(cmap(i)))
                # for i in range(num):
                #     each = lines_2d[i]
                #     x1, y1, x2, y2 = np.int0(each[0])
                #     color = rgba2rgb(cmap(i))
                #  drawn = cv2.line(src_for_draw, (x1, y1), (x2, y2), color, thickness=2)
                # <-------------------- revised for colored lines ---------------------->
                
                return lines_2d, lines_reshape.reshape((num, 1, 6)), num, drawn
            else:
                return lines_2d, lines_reshape.reshape((num, 1, 6)), num

    def draw_line_segment(self, _image, lines_2d, color=(255, 0, 0), thickness=1):
        """
        Draw line segments on the src image
        """
        for each in lines_2d:
            x1, y1, x2, y2 = np.int0(each[0])
            _image = cv2.line(_image, (x1, y1), (x2, y2), color, thickness)
        return _image
            
class TopoCoor(object):
    def __init__(self):
        pass

    def cross_product(self, a, b):
        x1, y1, z1 = a
        x2, y2, z2 = b
        return np.array([(y1 * z2 - y2 * z1), -(x1 * z2 - x2 * z1), (x1 * y2 - x2 * y1)])

    def gli(self, line1, line2):
        """
        Calculate Gaussian link integral in a geometrical way
        Parameters: 
            line1 {list} -- (p1.x,p1.y,p1.z, p2.x,p2.y,p2.z)
            line2 {list} -- (p1.x,p1.y,p1.z, p2.x,p2.y,p2.z)
        Returns: 
            w {float} -- writhe, sun of gli for all segment pairs
        """
        a = np.array([line1[0], line1[1], line1[2]])
        b = np.array([line1[3], line1[4], line1[5]])

        c = np.array([line2[0], line2[1], line2[2]])
        d = np.array([line2[3], line2[4], line2[5]])

        r_ac = c - a
        r_ad = d - a
        r_bc = c - b
        r_bd = d - b

        c_ac_ad = self.cross_product(r_ac, r_ad)
        c_ad_bd = self.cross_product(r_ad, r_bd)
        c_bd_bc = self.cross_product(r_bd, r_bc)
        c_bc_ac = self.cross_product(r_bc, r_ac)

        n_a = c_ac_ad / np.linalg.norm(c_ac_ad)
        n_b = c_ad_bd / np.linalg.norm(c_ad_bd)
        n_c = c_bd_bc / np.linalg.norm(c_bd_bc)
        n_d = c_bc_ac / np.linalg.norm(c_bc_ac)

        w = np.arcsin(np.dot(n_a, n_b)) + np.arcsin(np.dot(n_b, n_c)) + np.arcsin(np.dot(n_c, n_d)) + np.arcsin(
            np.dot(n_d, n_a))
        return np.nan_to_num(w)
    
    def topo_coor_from_edges(self, lines_3d):
        """
        Given lines segments from edges, output the topology coordinates
        Parameters:
            lines_3d {array} -- (num, 1, 6) [p1.x,p1.y,p1.z, p2.x,p2.y,p2.z]
        Return:
            writhe_matrix {array} -- (line num x line num)
            writhe {float} -- writhe
            density {float} -- density
        """
        lines_num = lines_3d.shape[0]
        writhe_matrix = np.zeros([lines_num, lines_num])
        wm_flatten = np.array([])
        for i in range(lines_num):
            for j in range(i + 1, lines_num):
                p0, p1, p2, p3, p4, p5 = lines_3d[i][0]
                q0, q1, q2, q3, q4, q5 = lines_3d[j][0]
                l1 = [p0, p1, p2, p3, p4, p5]
                l2 = [q0, q1, q2, q3, q4, q5]
                writhe = self.gli_original(l1, l2)
                writhe_matrix[i][j] = writhe
                wm_flatten = np.append(wm_flatten, writhe)

        # start computing writhe(avg.)
        writhe = np.sum(writhe_matrix) # we try to use the total writhe of the matrix
        # start computing density
        density_thre = np.mean(wm_flatten)
        if len(wm_flatten):
            density = len(wm_flatten[wm_flatten >= density_thre]) / len(wm_flatten)
        else:
            density = 0
        
        return writhe_matrix, writhe, density

    def topo_coor_from_two_curves(self, line1, line2):
        """
        Given two curves composed by several segments, output the topology coordinates
        * Two curves must have the same segment number
        Parameters:
            lines1 {array} -- (num, 1, 6) [p1.x,p1.y,p1.z, p2.x,p2.y,p2.z]
            lines2 {array} -- (num, 1, 6) [p1.x,p1.y,p1.z, p2.x,p2.y,p2.z]
        Return:
            writhe_matrix {array} -- (line num x line num)
            writhe {float} -- writhe
            density {float} -- density
        """
        lnum1 = line1.shape[0]
        lnum2 = line2.shape[0]
        writhe_matrix = np.zeros([lnum1, lnum2])
        wm_flatten = np.array([])

        for i in range(lnum1):
            for j in range(lnum2):
                p0, p1, p2, p3, p4, p5 = line1[i][0]
                q0, q1, q2, q3, q4, q5 = line2[j][0]
                l1 = [p0, p1, p2, p3, p4, p5]
                l2 = [q0, q1, q2, q3, q4, q5]
                writhe = self.gli_original(l1, l2)
                writhe_matrix[i][j] = writhe
                wm_flatten = np.append(wm_flatten, writhe)
        writhe = np.sum(writhe_matrix) # we try to use the total writhe of the matrix
        # start computing density
        density_thre = np.mean(wm_flatten)
        if len(wm_flatten):
            density = len(wm_flatten[wm_flatten >= density_thre]) / len(wm_flatten)
        else:
            density = 0

        return writhe_matrix, writhe, density

    def topo_coor_from_img(self, src, len_thld, dist_thld, cmask=False):
        """
        Given an image, output the topology coordinates
        Parameters:
            src {array} -- 3-channel image
        Return:
            writhe_matrix {array} -- (line num x line num)
            writhe {float} -- writhe
            density {float} -- density
            center_mask {array} -- (same size as src) if_center_mask is True
        """
        ld = LineDetection()
        lines_2d, lines_3d, lines_num = ld.detect_line(src, len_thld, dist_thld)
        if lines_num is None:
            if cmask is False:
                return None, None ,None
            else:
                return None, None ,None, None

        else:
            writhe_matrix = np.zeros([lines_num, lines_num])
            wm_flatten = np.array([])
            for i in range(lines_num):
                for j in range(i + 1, lines_num):
                    p0, p1, p2, p3, p4, p5 = lines_3d[i][0]
                    q0, q1, q2, q3, q4, q5 = lines_3d[j][0]
                    l1 = [p0, p1, p2, p3, p4, p5]
                    l2 = [q0, q1, q2, q3, q4, q5]
                    writhe = self.gli(l1, l2)
                    writhe_matrix[i][j] = writhe
                    wm_flatten = np.append(wm_flatten, writhe)
            
            # start computing writhe(avg.)
            writhe = np.sum(writhe_matrix) / lines_num  # here we use the average of writhe matrix

            # start computing density
            DENSITY_THRE = np.mean(wm_flatten)
            if len(wm_flatten):
                density = len(wm_flatten[wm_flatten >= DENSITY_THRE]) / len(wm_flatten)
            else:
                density = 0 

            if cmask is False: return writhe_matrix, writhe, density
            else:
                # start computing center
                (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(writhe_matrix)
                center = maxLoc
                # center = np.int0(ndimage.center_of_mass(writhe_matrix))

                # generate center mask
                c1 = lines_2d[center[0]][0]
                c2 = lines_2d[center[1]][0]
                center_mask = np.zeros((src.shape[0], src.shape[1]))
                # (x1,y1): left-top, (x2,y2):right-bottom
                x1 = int(np.min([c1[0], c1[2], c2[0], c2[2]]))
                x2 = int(np.max([c1[0], c1[2], c2[0], c2[2]]))
                y1 = int(np.min([c1[1], c1[3], c2[1], c2[3]]))
                y2 = int(np.max([c1[1], c1[3], c2[1], c2[3]]))
                # draw center for test
                # test = cv2.line(src, (int(c1[0]), int(c1[1])), (int(c1[2]), int(c1[3])), (0, 255.0), thickness=3)
                # test = cv2.line(test, (int(c2[0]), int(c2[1])), (int(c2[2]), int(c2[3])), (0, 255, 0), thickness=3)
                center_mask[y1:y2, x1:x2] = 255
                # cv2.imwrite("./DEPTH_TMP/center_line.png", test)
                # cv2.imwrite("./DEPTH_TMP/center_mask.png", center_mask)
                return writhe_matrix, writhe, density, center_mask
