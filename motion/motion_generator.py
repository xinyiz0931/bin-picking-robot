import os
import sys
sys.path.append("./")
import cv2
import numpy as np
import math
from utils.base_utils import *


class Motion(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.initialpose = "0 10 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -133.7 -7 0 0 0 0 0 0"
        self.placepose = "0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0"

    def empty_motion_generator(self):
        open(self.filepath, 'w').close()
    
    def motion_generator_basic(self, rx,ry,rz,ra):
        fp = open(self.filepath, 'wt')
        print("0 1 LHAND_JNT_OPEN",file=fp)
        print("0 1.5 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)

        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 1 LARM_XYZ_ABS 0.5 0.29 0.3 -180 -90 145",file=fp)
        # place
        print("0 0.5 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()

    def motion_generator_dl(self, rx,ry,rz,ra):
        """Action for directly lifting"""
        fp = open(self.filepath, 'wt')
        print("0 1 LHAND_JNT_OPEN",file=fp)
        print("0 1.5 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)

        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        # print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 1.5 LARM_XYZ_ABS 0.5 0.23 0.40 -180 -90 145",file=fp)
        # place
        print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()




    def motion_generator_half(self, rx,ry,rz,ra):
        """full Circle-like motion"""
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
        print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
        print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
        print("0 2 JOINT_REL 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()

    def motion_generator_half_spin(self, rx,ry,rz,ra):
        """full Circle-like motion with spinning"""
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
        print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
        print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)

        print("0 0.75 LARM_JNT_REL 0 0 0 0 0 0 120",file=fp)
        print("0 0.75 LARM_JNT_REL 0 0 0 0 0 0 -100",file=fp)

        print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
        # print("0 0.75 LARM_JNT_REL 0 0 0 0 0 0 120",file=fp)
        # print("0 0.75 LARM_JNT_REL 0 0 0 0 0 0 -100",file=fp)
        print("0 2 JOINT_REL 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()
    
    def motion_generator_full(self, rx,ry,rz,ra):
        """half Circle-like motion"""
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.53 0.21 0.37 155 -73 -176",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.580 0.01 0.42 160 -57 170",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.534 -0.18 0.43 174 -45 144",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131",file=fp)
        print("0 1 LARM_XYZ_ABS 0.43 0.24 0.4 -180 -90 145",file=fp)
        print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()
        # fp = open(self.filepath, 'wt')
        # print("0 0.5 LHAND_JNT_OPEN",file=fp)
        # print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        # print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        # print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        # print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        # print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        # print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        # print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL -0.05 0.15 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
        # print("0 0.75 LARM_XYZ_REL 0.06 0.06 0 0 0 0",file=fp)
        # print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        # print("0 0.5 LHAND_JNT_OPEN",file=fp)
        # print(self.initialpose,file=fp)
        # fp.close()
    
    def motion_generator_full_spin(self, rx,ry,rz,ra):
        """Full circle-like motion with spinning"""
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.53 0.21 0.37 155 -73 -176",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.580 0.01 0.42 160 -57 170",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.534 -0.18 0.43 174 -45 144",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131",file=fp)
        print("0 1 LARM_XYZ_ABS 0.43 0.24 0.4 -180 -90 145",file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 120",file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 -150",file=fp)
        print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()
    

    

    def motion_generator_two_full(self, rx,ry,rz,ra):
        """two Circle-like motion"""
        # fp = open(self.filepath, 'wt')
        # print("0 0.5 LHAND_JNT_OPEN",file=fp)
        # print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        # print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        # print("0 1 LARM_XYZ_REL 0 0 -0.1 0 0 0",file=fp)
        # print("0 1.2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.2),file=fp)
        # print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        # print("0 1.2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        # print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)

        # print("0 0.5 LARM_XYZ_ABS 0.53 0.21 0.37 155 -73 -176",file=fp)
        # print("0 0.75 LARM_XYZ_ABS 0.580 0.01 0.42 160 -57 170",file=fp)
        # print("0 0.75 LARM_XYZ_ABS 0.534 -0.18 0.43 174 -45 144",file=fp)
        # print("0 0.75 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131",file=fp)

        # print("0 1 LARM_XYZ_ABS 0.43 0.16 0.46 -180 -90 145",file=fp)
        # print("0 1 LARM_XYZ_ABS 0.537 0.160 0.47 -180 -90 145",file=fp)
        # print("0 0.5 LARM_XYZ_ABS 0.597 0.06 0.47 -180 -90 145",file=fp)
        # print("0 0.5 LARM_XYZ_ABS 0.627 -0.14 0.48 -180 -90 145",file=fp)
        # print("0 0.5 LARM_XYZ_ABS 0.527 -0.29 0.48 -180 -90 145",file=fp)
        # print("0 1 LARM_XYZ_ABS 0.477 -0.14 0.47 -180 -90 145",file=fp)
        # print("0 1 LARM_XYZ_ABS 0.457 0.16 0.46 -180 -90 145",file=fp)
        """two Circle-like motion"""
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.1 0 0 0",file=fp)
        print("0 1.2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.2),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1.2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        
        
        print("0 0.5 LARM_XYZ_ABS 0.597 0.06 0.34 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.627 -0.14 0.36 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.527 -0.29 0.38 -180 -90 145",file=fp)
        print("0 1 LARM_XYZ_ABS 0.477 -0.14 0.40 -180 -90 145",file=fp)
        print("0 1 LARM_XYZ_ABS 0.457 0.16 0.41 -180 -90 145",file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.42 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.53 0.21 0.43 155 -73 -176",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.580 0.01 0.44 160 -57 170",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.534 -0.18 0.44 174 -45 144",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131",file=fp)
        print("0 1 LARM_XYZ_ABS 0.43 0.16 0.46 -180 -90 145",file=fp)
        

        print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()
        # fp = open(self.filepath, 'wt')
        # print("0 0.5 LHAND_JNT_OPEN",file=fp)
        # print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        # print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        # print("0 1 LARM_XYZ_REL 0 0 -0.1 0 0 0",file=fp)
        # print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.2),file=fp)
        # print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        # print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        # print("0 1.5 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        # print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL -0.05 0.15 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL 0.06 0.06 -0.02 0 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0 0 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL -0.05 -0.2 0.02 0 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL -0.05 0.15 0.02 0 0 0",file=fp)
        # print("0 0.5 LARM_XYZ_REL -0.02 0.15 0.02 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL 0.06 0.06 0 0 0 0",file=fp)
        # print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        # print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        # print("0 0.5 LHAND_JNT_OPEN",file=fp)
        # print(self.initialpose,file=fp)
        # fp.close()

    def motion_generator_two_full_spin(self, rx,ry,rz,ra):
        """two Circle-like motion"""
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.1 0 0 0",file=fp)
        print("0 1.2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.2),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1.2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
        
        print("0 0.5 LARM_XYZ_ABS 0.597 0.06 0.34 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.627 -0.14 0.36 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.527 -0.29 0.38 -180 -90 145",file=fp)
        print("0 1 LARM_XYZ_ABS 0.477 -0.14 0.40 -180 -90 145",file=fp)
        print("0 1 LARM_XYZ_ABS 0.457 0.16 0.41 -180 -90 145",file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.42 -180 -90 145",file=fp)
        print("0 0.5 LARM_XYZ_ABS 0.53 0.21 0.43 155 -73 -176",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.580 0.01 0.44 160 -57 170",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.534 -0.18 0.44 174 -45 144",file=fp)
        print("0 0.75 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131",file=fp)
        print("0 1 LARM_XYZ_ABS 0.43 0.16 0.46 -180 -90 145",file=fp)
        

        print("0 0.75 LARM_JNT_REL 0 0 0 0 0 0 150",file=fp)
        print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()

    def motion_generator_spin(self, rx,ry,rz,ra):
        """Spin motion"""
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
        print("0 1 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
        print("0 2 LARM_XYZ_ABS 0.52 0.05 0.3 -180 -90 145",file=fp)
        print("0 2 LARM_JNT_REL 0 0 0 0 0 0 320",file=fp)
        # print("0 2 LARM_XYZ_ABS 0.5 0.23 0.3 -180 -90 145",file=fp)
        print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 2 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
        print("0 1 LHAND_JNT_OPEN",file=fp)
        print("0 2 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
        print(self.initialpose,file=fp)
        fp.close()

    def motion_generator_just_pick(self, rx,ry,rz,ra):
        fp = open(self.filepath, 'wt')
        print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)

        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
        print("0 1 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
        print("1 PAUSE")
        fp.close()
    
    def motion_generator_finish(self):
        fp = open(self.filepath,'a')
        fp.write("0 2 LARM_XYZ_ABS 0.5 0.15 0.45 -180 -90 145\n") # place action old y = 0.2
        fp.write("0 1 LARM_XYZ_REL 0 0 -0.20 0 0 0\n")
        fp.write("0 1 LHAND_JNT_OPEN\n")
        fp.write("0 1 LARM_XYZ_REL 0 0 0.20 0 0 0\n")
        fp.write("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 0 -33.7 -122.7 0 0 0 0 0 0 0\n")
        fp.close()
        


    


# class Motion(object):
#     def __init__(self, imgpath, filepath, max_h, min_h):
#         self.imgpath = imgpath
#         self.filepath = filepath
#         self.max_h = max_h
#         self.min_h = min_h

#     def get_neighbor_pixel(gray,loc,bounding_size=10):
#         (x,y) = loc
#         h,w = gray.shape
#         mat = gray[(y-bounding_size):(y+bounding_size+1),(x-bounding_size):(x+bounding_size+1)]
#         left_margin = x-bounding_size
#         top_margin = y-bounding_size

#         # index = w * Y + X
#         max_xy = np.where(mat == mat.max())

#         y_p = max_xy[0][0] + y-bounding_size
#         x_p = max_xy[1][0] + x-bounding_size
#         return (x_p, y_p),mat.max()
        

    
#     def motion_generator(self, rx,ry,rz,ra):
#         fp = open(self.filepath, 'wt')
#         print("0 1 JOINT_ABS 0 0 0 8 -30.7 -127.5 -10 0 0 0 -35.7 -127.5 0 0 0 0 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)

#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
#         print("0 1 LHAND_JNT_CLOSE",file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.3 - rz),file=fp)
#         print("0 1 LARM_XYZ_ABS 0.5 0.23 0.3 -180 -90 145",file=fp)
#         # place
#         # print("0 2 LARM_XYZ_ABS 0.3301 0.4404 0.3 52 -90 -58",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
#         print("0 1 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         fp.close()
#         return 

#     # def motion_generator(self, rx,ry,rz,ra):
#     #     """EASY VERSION"""
#     #     fp = open(self.filepath, 'wt')
#     #     print("0 2 JOINT_ABS 0 0 0 8 -30.7 -127.5 -10 0 0 0 -35.7 -127.5 0 0 0 0 0 0 0",file=fp)
#     #     print("0 0.5 LHAND_JNT_OPEN",file=fp)
#     #     print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
#     #     print("0 1 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#     #     fp.close()
#     #     return 

#     def motion_generator_just_pick(self, rx,ry,rz,ra):
#         fp = open(self.filepath, 'wt')
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)

#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
#         print("0 1 LHAND_JNT_CLOSE",file=fp)
#         print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
#         print("1 PAUSE")
#         fp.close()
    
#     def motion_generator_finish(self):
#         fp = open(self.filepath,'a')
#         fp.write("0 2 LARM_XYZ_ABS 0.5 0.15 0.45 -180 -90 145\n") # place action old y = 0.2
#         fp.write("0 1 LARM_XYZ_REL 0 0 -0.20 0 0 0\n")
#         fp.write("0 1 LHAND_JNT_OPEN\n")
#         fp.write("0 1 LARM_XYZ_REL 0 0 0.20 0 0 0\n")
#         fp.write("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 0 -33.7 -122.7 0 0 0 0 0 0 0\n")
#         fp.close()
        
#     def motion_generator_trcj1(self, rx,ry,rz,ra):
#         """Circle-like motion"""
#         fp = open(self.filepath, 'wt')
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
#         print("0 1 LHAND_JNT_CLOSE",file=fp)
#         print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
#         print("0 1.5 LARM_XYZ_ABS 0.537 0.160 0.3 -180 -90 145",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.05 -0.1 0.05 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.05 -0.14 0.02 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.05 -0.13 0.03 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.11 -0.1 0.03 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0.05 0.03 0 0 0",file=fp)
#         print("0 2 LARM_XYZ_ABS 0.5 0.23 0.3 -180 -90 145",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         fp.close()

#     def motion_generator_trcj2(self, rx,ry,rz,ra):
#         """Circle-like motion"""
#         fp = open(self.filepath, 'wt')
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
#         print("0 1 LHAND_JNT_CLOSE",file=fp)
#         print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
#         print("0 1.5 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL -0.05 0.15 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL 0.06 0.06 -0.02 0 0 0",file=fp)
#         # print("0 1 LARM_XYZ_REL -0.1 -0.1 0 0 0 0",file=fp)
#         # print("0 1 LARM_XYZ_REL -0.1 0.28 0.03 0 0 0",file=fp)
#         print("0 2 LARM_XYZ_ABS 0.5 0.23 0.3 -180 -90 145",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         fp.close()
        
#     def motion_generator_trcj3(self, rx,ry,rz,ra):
#         """Circle-like motion"""
#         fp = open(self.filepath, 'wt')
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
#         print("0 1 LHAND_JNT_CLOSE",file=fp)
#         print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
#         print("0 1.5 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL -0.05 0.15 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL 0.06 0.06 -0.02 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.05 -0.2 0.02 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.05 0.15 0.02 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.02 0.15 0.02 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL 0.06 0.06 0 0 0 0",file=fp)
#         # print("0 1 LARM_XYZ_REL -0.1 -0.1 0 0 0 0",file=fp)
#         # print("0 1 LARM_XYZ_REL -0.1 0.28 0.03 0 0 0",file=fp)
#         print("0 2 LARM_XYZ_ABS 0.5 0.23 0.3 -180 -90 145",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         fp.close()

#     def motion_generator_trcj3_ex(self, rx,ry,rz,ra):
#         """Spin motion"""
#         fp = open(self.filepath, 'wt')
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
#         print("0 1 LHAND_JNT_CLOSE",file=fp)
#         print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
#         print("0 2 LARM_XYZ_ABS 0.52 0.05 0.3 -180 -90 145",file=fp)
#         print("0 2 LARM_JNT_REL 0 0 0 0 0 0 320",file=fp)
#         print("0 2 LARM_XYZ_ABS 0.5 0.23 0.3 -180 -90 145",file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
#         print("0 1 LHAND_JNT_OPEN",file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         fp.close()
    
#     def motion_generator_trcj4(self, rx,ry,rz,ra):
#         """Circle-like motion"""
#         fp = open(self.filepath, 'wt')
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
#         print("0 2 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
#         print("0 1 LHAND_JNT_CLOSE",file=fp)
#         print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.33 - rz),file=fp)
#         print("0 1.5 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
#         print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL -0.05 0.15 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_REL 0.06 0.06 -0.02 0 0 0",file=fp)
#         print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 120",file=fp)
#         # print("0 1 LARM_XYZ_REL -0.1 -0.1 0 0 0 0",file=fp)
#         # print("0 1 LARM_XYZ_REL -0.1 0.28 0.03 0 0 0",file=fp)
#         print("0 1 LARM_XYZ_ABS 0.5 0.23 0.3 -180 -90 145",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
#         print("0 0.5 LHAND_JNT_OPEN",file=fp)
#         print("0 1 LARM_XYZ_REL 0 0 0.10 0 0 0",file=fp)
#         print("0 2 JOINT_ABS 0 0 0 0 -25.7 -127.5 0 0 0 8 -25.7 -133.7 -7 0 0 0 0 0 0",file=fp)
#         fp.close()



def rotate_img(img, angle, center=None, scale=1.0):
    (h,w) = img.shape[:2]

    if center is None:
        center=(w/2, h/2)

    M = cv2.getRotationMatrix2D(center, angle,scale)
    rotated = cv2.warpAffine(img, M, (w,h))
    return rotated
    

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("./")
    img_path = os.path.join(ROOT_DIR, "depth/MedianFilterDepth_Rotated.png")
    import timeit
    start = timeit.default_timer()
    img = cv2.imread(img_path)
    for r in range(0,180,45):
        rotate_img(img, r)
    stop = timeit.default_timer()
    print('[OpenCV] Time: ', stop - start)  

    start = timeit.default_timer()
    from PIL import Image
    # img = Image.open(img_path)
    img = cv2.imread(img_path)
    img = Image.fromarray(img)
    for r in range(0,180,45):
        img.rotate(r)
    stop = timeit.default_timer()
    print('[PIL] Time: ', stop - start)  

    """
    

    """
   
