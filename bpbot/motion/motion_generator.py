from http.client import REQUEST_URI_TOO_LONG
import os
from re import I
import cv2
import numpy as np
import math

class Motion(object):
    def __init__(self, filepath):
        self.filepath = filepath
        self.initialpose = "0 3 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -133.7 -7 0 0 0 0 0 0"
        self.initialpose_ = "0 3 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -133.7 -7 0 0 0.0300 -0.0300 0.0240 -0.0240"
        
        self.placepose = [
            "0 2 JOINT_REL 80 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",
            "0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0", 
            "0 1 LHAND_JNT_OPEN" 
        ]
        self.half_helix = [
            "0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145", 
            "0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",  
            "0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0", 
            "0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0", 
            "0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0"
        ]
        self.helix = [
            "0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145", # o
            "0 0.5 LARM_XYZ_ABS 0.6 0.12 0.34 -180 -90 145", # half
            "0 0.5 LARM_XYZ_ABS 0.627 -0.14 0.36 -180 -90 145", # half
            "0 0.5 LARM_XYZ_ABS 0.537 -0.32 0.38 -180 -90 145", # full
            "0 0.75 LARM_XYZ_ABS 0.490 -0.18 0.40 -180 -90 145", # full
            "0 0.75 LARM_XYZ_ABS 0.437 0.08 0.41 -180 -90 145", # full
            "0 0.75 LARM_XYZ_ABS 0.537 0.260 0.42 -180 -90 145", # o
            "0 0.5 LARM_XYZ_ABS 0.557 0.21 0.43 155 -73 -176", # two full
            "0 0.55 LARM_XYZ_ABS 0.580 0.01 0.44 160 -57 170", # two full
            "0 0.55 LARM_XYZ_ABS 0.534 -0.18 0.44 174 -45 144", # two full
            "0 0.55 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131", # two full
            "0 1 LARM_XYZ_ABS 0.43 0.26 0.46 -180 -90 145" # two full
        ]
        self.helix_slow = [
            "0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145", # o
            "0 3 LARM_XYZ_ABS 0.6 0.12 0.34 -180 -90 145", # half
            "0 3 LARM_XYZ_ABS 0.627 -0.14 0.36 -180 -90 145", # half
            "0 3 LARM_XYZ_ABS 0.537 -0.32 0.38 -180 -90 145", # full
            "0 3 LARM_XYZ_ABS 0.490 -0.18 0.40 -180 -90 145", # full
            "0 3 LARM_XYZ_ABS 0.437 0.08 0.41 -180 -90 145", # full
            "0 3 LARM_XYZ_ABS 0.537 0.260 0.42 -180 -90 145", # o
            "0 3 LARM_XYZ_ABS 0.557 0.21 0.43 155 -73 -176", # two full
            "0 3 LARM_XYZ_ABS 0.580 0.01 0.44 160 -57 170", # two full
            "0 1 LARM_XYZ_ABS 0.534 -0.18 0.44 174 -45 144", # two full
            "0 1 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131", # two full
            "0 1 LARM_XYZ_ABS 0.43 0.26 0.46 -180 -90 145" # two full
        ]
        
        self.diagnal = [
            "0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145", 
            # "0 1.5 LARM_XYZ_ABS 0.547 -0.29 0.46 -180 -90 145",
            "0 1.5 LARM_XYZ_ABS 0.537 -0.29 0.46 120 -90 145",
            "0 1.5 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145" 
        ]
        self.helix_cone = [
            "0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145", # o
            "0 0.5 LARM_XYZ_ABS 0.567 0.12 0.34 -180 -90 145", # half
            "0 0.5 LARM_XYZ_ABS 0.587 -0.04 0.36 -180 -90 145", # half
            "0 0.5 LARM_XYZ_ABS 0.527 -0.12 0.38 -180 -90 145", # full
            "0 0.75 LARM_XYZ_ABS 0.490 -0.08 0.40 -180 -90 145", # full
            "0 0.75 LARM_XYZ_ABS 0.437 0.08 0.41 -180 -90 145", # full
            "0 0.75 LARM_XYZ_ABS 0.507 0.160 0.42 -180 -90 145", # o
            "0 0.5 LARM_XYZ_ABS 0.527 0.12 0.43 155 -73 -176", # two full
            "0 0.55 LARM_XYZ_ABS 0.46 0.01 0.44 160 -57 170", # two full
            "0 0.55 LARM_XYZ_ABS 0.43 -0.03 0.44 174 -45 144", # two full
            "0 0.55 LARM_XYZ_ABS 0.4 -0.05 0.44 -168 -63 131", # two full
            "0 1 LARM_XYZ_ABS 0.43 0.26 0.46 -180 -90 145" # two full
        ]

        self.spin = ["0 0.5 LARM_JNT_REL 0 0 0 0 0 0 150", 
                     "0 0.5 LARM_JNT_REL 0 0 0 0 0 0 -150"]
    
    def gen_pickandplace_motion(self, rx,ry,rz,ra, dest='goal'):
        # ra is degree
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        # print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.2 {:.3f} -90 0".format(rx,ry,ra),file=fp)

        # print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.2),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        # place
        if dest == 'goal':
            print("0 1 LARM_XYZ_REL 0 0 {:.3f} {:.3f} 0 0".format(0.3-rz, -ra),file=fp)
            # print("0 0.8 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.3 - rz),file=fp)
            # print("0 0.5 LARM_XYZ_ABS 0.5 0.20 0.3 -180 -90 145",file=fp)
            # print("0 1 JOINT_REL 59 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0", file=fp)
            print("0 1 LARM_XYZ_ABS 0.0861 0.5316 0.3 55.8 -90 -31.8", file=fp)
            print("0 0.5 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        elif dest == 'mid':
            print("0 1 LARM_XYZ_REL 0 0 {:.3f} {:.3f} 0 0".format(0.4-rz, -ra),file=fp)
            # print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.40 - rz),file=fp)
            print("0 1 LARM_XYZ_ABS 0.48 0.240 0.4 -180 -90 145",file=fp)
            print("0 1 LARM_XYZ_REL 0 0 -0.10 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()
    
    def gen_separation_motion(self, rx,ry,rz,ra,rvx,rvy, vlen=0.15):
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.3),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        # pull
        print("0 1.5 LARM_XYZ_REL {:.3f} {:.3f} 0 0 0 0".format(vlen*rvx, vlen*rvy), file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.3 - rz),file=fp)
        # place
        print("0 1 LARM_XYZ_ABS 0.0861 0.5316 0.3 55.8 -90 -31.8", file=fp)
        # print("0 1 LARM_XYZ_ABS 0.5 0.180 0.3 -180 -90 145",file=fp)
        # print("0 1 JOINT_REL 50 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0", file=fp)
        # print("0 2 JOINT_ABS 0.1032 0.5213, 0.3000 58.2 -90 -34.2", file=fp)
        print("0 0.5 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()
     
    def gen_separation_motion_dualarm(self,g_hold,g_pull,v_pull,len=0.05):
        [hx,hy,hz,hroll,hpitch,hyaw] = g_hold
        [px,py,pz,proll,ppitch,pyaw] = g_pull
        [vx,vy] = v_pull
        len = 0.1
        fp = open(self.filepath, 'wt')
        #print(self.initialpose_,file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 0.5 RHAND_JNT_OPEN",file=fp)
        print("0 3 LARM_XYZ_ABS {:.3f} {:.3f} 0.2 {:.1f} {:.1f} {:.1f}".format(px,py,proll,ppitch,pyaw),file=fp)
        print("0 3 LARM_XYZ_ABS {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f}".format(px,py,pz,proll,ppitch,pyaw),file=fp)

        # print("0 3 RARM_XYZ_ABS {:.3f} {:.3f} {:.3f} {:.3f} -90 0".format(hx,hy,hz,ha),file=fp)
        print("0 3 RARM_XYZ_ABS {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f}".format(hx,hy,hz,hroll,hpitch,hyaw),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 0.5 RHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)

        print("0 3 LARM_XYZ_ABS {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f}".format(px+len*vx,py+len*vy,pz+0.02,proll,ppitch,pyaw),file=fp)
        print("0 3 LARM_XYZ_ABS {:.3f} {:.3f} {:.3f} {:.1f} {:.1f} {:.1f}".format(px+len*vx,py+len*vy,pz+0.1,proll,ppitch,pyaw),file=fp)

        # place
        print("0 2 LARM_XYZ_ABS 0.48 0.35 0.25 {:.3f} {:.1f} {:.1f}".format(proll,ppitch,pyaw),file=fp) 
        # print("0 1 LARM_XYZ_ABS 0.0861 0.5316 0.3 55.8 -90 -31.8", file=fp)
        # print("0 1 LARM_XYZ_ABS 0.0861 0.5316 0.12 55.8 -90 -31.8", file=fp)

        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 0.5 RHAND_JNT_OPEN",file=fp)
        print(self.initialpose_,file=fp)
        fp.close()
     
    def get_pick_pose(self, rx, ry, rz, ra):
        return [
            "0 0.5 LHAND_JNT_OPEN", 
            "0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),
            "0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),
            "0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",
            "0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),
            "0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",
            "0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),
            "0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",
        ]

    def empty_motion_generator(self):
        open(self.filepath, 'w').close()
    
    def generate_cone_helix(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        # pick
        for pick_m in self.get_pick_pose(rx, ry, rz, ra):
            print(pick_m, file=fp)
        # separate
        for sep_m in self.helix_cone:
            print(sep_m, file=fp)
        # place 
        for place_m in self.placepose:
            print(place_m, file=fp)
        print(self.initialpose, file=fp)
        fp.close()
    def generate_cone_helix_spin(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        # pick
        for pick_m in self.get_pick_pose(rx, ry, rz, ra):
            print(pick_m, file=fp)
        # separate
        for sep_m in self.helix_cone:
            print(sep_m, file=fp)
        for spin_m in self.spin:
            print(spin_m, file=fp)
        # place 
        for place_m in self.placepose:
            print(place_m, file=fp)
        print(self.initialpose, file=fp)
        fp.close()
    def generate_diagnal(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        # pick
        for pick_m in self.get_pick_pose(rx,ry,rz,ra):
            print(pick_m, file=fp)
        # separate
        for sep_m in self.diagnal:
            print(sep_m, file=fp)
        for place_m in self.placepose:
            print(place_m, file=fp)
        print(self.initialpose, file=fp)
        fp.close()

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
        # helix 
        print("0 0.5 LARM_XYZ_REL 0.06 -0.1 0.05 0 0 0",file=fp)
        print("0 0.5 LARM_XYZ_REL 0.03 -0.2 0.05 0.05 0 0",file=fp)
        print("0 0.5 LARM_XYZ_REL -0.1 -0.15 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL -0.02 0.35 0 0 0 0",file=fp)
        print("0 2 JOINT_REL 90 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()

    # def generate_a_h(self, rx, ry, rz, ra):
    #     fp = open(self.filepath, 'wt')
    #     print("0 0.5 LHAND_JNT_OPEN",file=fp)
    #     print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
    #     print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
    #     print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
    #     print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
    #     print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
    #     print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
    #     # helix 
    #     for h in self.half_helix: 
    #         print(h, file=fp)
    #     for p in self.placepose:
    #         print(p, file=p)
    #     print(self.initialpose, file=fp)
    #     fp.close()
    
    def generate_a_h(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp) 
        # helix 
        for h in self.helix[0:4]:
            print(h, file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.42 -180 -90 145",file=fp)
        for p in self.placepose:
            print(p, file=fp)
        print(self.initialpose, file=fp)
        fp.close()

    def generate_a_hs(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp) 
        # helix 
        for h in self.helix[0:4]:
            print(h, file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.42 -180 -90 145",file=fp)
        for s in self.spin:
            print(s, file=fp)

        for p in self.placepose:
            print(p, file=fp)
        print(self.initialpose, file=fp)
        fp.close()

    def generate_a_f(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp) 
        # helix 
        for h in self.helix[0:6]:
            print(h, file=fp)
        for p in self.placepose:
            print(p, file=fp)
        print(self.initialpose, file=fp)
        fp.close()

    def generate_a_fs(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp) 
        # helix 
        for h in self.helix[0:6]:
            print(h, file=fp)
        for s in self.spin:
            print(s, file=fp)
        for p in self.placepose:
            print(p, file=fp)
        print(self.initialpose, file=fp)
        fp.close()

    def generate_a_tf(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp) 
        # helix 
        # for h in self.helix:
        for h in self.helix_slow:
            print(h, file=fp)
        for p in self.placepose:
            print(p, file=fp)
        print(self.initialpose, file=fp)
        fp.close()

    def generate_a_tfs(self, rx, ry, rz, ra):
        fp = open(self.filepath, 'wt')
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
        print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
        print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
        print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp) 
        # helix 
        for h in self.helix:
            print(h, file=fp)
        for s in self.spin:
            print(s, file=fp)
        for p in self.placepose:
            print(p, file=fp)
        print(self.initialpose, file=fp)
        fp.close()

    def motion_generator_half_spin(self, rx,ry,rz,ra):
        """half Circle-like motion with spinning"""
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
        # print("0 1 LARM_XYZ_ABS 0.457 0.16 0.41 -180 -90 145",file=fp)
        print("0 1 LARM_XYZ_ABS 0.537 0.160 0.42 -180 -90 145",file=fp)
        # print("0 0.5 LARM_XYZ_ABS 0.53 0.21 0.43 155 -73 -176",file=fp)
        # print("0 0.75 LARM_XYZ_ABS 0.580 0.01 0.44 160 -57 170",file=fp)
        # print("0 0.75 LARM_XYZ_ABS 0.534 -0.18 0.44 174 -45 144",file=fp)
        # print("0 0.75 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131",file=fp)
        # print("0 1 LARM_XYZ_ABS 0.43 0.16 0.46 -180 -90 145",file=fp)
        

        print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
        print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
        print("0 0.5 LHAND_JNT_OPEN",file=fp)
        print(self.initialpose,file=fp)
        fp.close()     
    
    # def motion_generator_full(self, rx,ry,rz,ra):
    #     """Full circle-like motion"""
    #     fp = open(self.filepath, 'wt')
    #     print("0 0.5 LHAND_JNT_OPEN",file=fp)
    #     print("0 1 LARM_XYZ_ABS {:.3f} {:.3f} 0.3 -180 -90 145".format(rx,ry),file=fp)
    #     print("0 0.5 LARM_JNT_REL 0 0 0 0 0 0 {:.3f}".format(33.0 + ra),file=fp)
    #     print("0 1 LARM_XYZ_REL 0 0 -0.15 0 0 0",file=fp)
    #     print("0 1.5 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(rz - 0.15),file=fp)
    #     print("0 0.5 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
    #     print("0 1 LARM_XYZ_REL 0 0 {:.3f} 0 0 0".format(0.20 - rz),file=fp)
    #     print("0 1 LARM_XYZ_ABS 0.537 0.160 0.32 -180 -90 145",file=fp)
    #     print("0 0.5 LARM_XYZ_ABS 0.53 0.21 0.37 155 -73 -176",file=fp)
    #     print("0 0.75 LARM_XYZ_ABS 0.580 0.01 0.42 160 -57 170",file=fp)
    #     print("0 0.75 LARM_XYZ_ABS 0.534 -0.18 0.43 174 -45 144",file=fp)
    #     print("0 0.75 LARM_XYZ_ABS 0.43 -0.23 0.44 -168 -63 131",file=fp)
    #     print("0 1 LARM_XYZ_ABS 0.43 0.24 0.4 -180 -90 145",file=fp)
    #     print("0 2 JOINT_REL 70 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0",file=fp)
    #     print("0 1 LARM_XYZ_REL 0 0 -0.25 0 0 0",file=fp)
    #     print("0 0.5 LHAND_JNT_OPEN",file=fp)
    #     print(self.initialpose,file=fp)
    #     fp.close()

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

   
