import os
import re
import cv2
import argparse
import numpy as np
import importlib
sepc = importlib.util.find_spec("cnoid")
found_cnoid = sepc is not None
if found_cnoid:
    from cnoid.Util import *
    from cnoid.Base import *
    from cnoid.Body import *
    from cnoid.BodyPlugin import *
    from cnoid.GraspPlugin import *
    from cnoid.BinPicking import *

from bpbot.robotcon import NxtRobot
from bpbot.driver import PhxClient

def main():
    parser = argparse.ArgumentParser('Hand eye calibration tool')
    parser.add_argument('-m', '--mode', choices=["calib", "calc", "vis"], required=True)
    parser.add_argument('-f', '--file', help='calibration result folder')

    args = parser.parse_args()

class HandEye(object):
    def __init__(self, mode, folder):
        self.mode = mode

        calib_dir = os.path.realpath(folder)
        save_hand = os.path.join(calib_dir, "pos_hand.txt")
        save_eye = os.path.join(calib_dir, "pos_eye.txt")

        mf_path = os.path.join(calib_dir, "calib_3d.dat")
        pre_robot = os.path.join(calib_dir, "robot.txt")
        
        self.mkid = 7
        self.fix_waist = False
        self.pos_eye = []
        self.pos_hand = []
    
        if self.mode == "calib" and found_cnoid:
            self.robot = NxtRobot(host='[::]:15005')

    def extract_number(self, string):
        # return [int(s) for s in re.findall(r'\d+(?:\e\d+)?', string)] 
        return re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)

    def extract_motionfile(self, filepath):
        """Extract robot pose (xyz+rpy) from motion file

        Args:
            filepath (str): path to motion file
        Returns:
            pose (array): shape=(N,6), extracted poses
        """
        xyz = []
        with open(filepath, 'r+') as fp:
            lines = fp.readlines()
        print(lines)
        for line in lines:
            if "#" in line: # comment in motion file 
                continue
            # print(line.split(' '))
            print(self.extract_number(line))
            xyz.append([float(x) for x in self.extract_number(line)[2:5]])
        print(xyz)

print("Hand eye calibration")
actor = HandEye("calib", "./data\\calibration\\test")
actor.extract_motionfile(filepath="C:\\Users\\xinyi\\Code\\bpbot\\data\\calibration\\test\\calib_3d.dat")