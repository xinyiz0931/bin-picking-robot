import os
import sys
sys.path.append("./")
import numpy as np
import argparse
from robotcon.nxt.nxtrobot_client import NxtRobot
from utils.base_utils import *

if __name__ == "__main__":
    # absolute path needed
    mfik_path =  "/home/xinyi/Workspace/myrobot/motion/motion_ik.dat"
    parser = argparse.ArgumentParser(description='ik file path')
    parser.add_argument('--filepath','-f', type=str, 
                        help='ik file path', default=mfik_path)
    args = parser.parse_args()

    motion_seq = np.loadtxt(args.filepath)
    nxt = NxtRobot(host='[::]:15005')

    main_proc_print("Start robot execution .. ")
    """
    each line for mfik: 
    m[0]: motion time (1)
    m[1]: hand action(1)
        0: close
        1: open
        2: stay
        3: pause flag
    m[2:17]: whole body joint (15)
    m[18:21]: gripper control (4)
    """

    for m in motion_seq:
        if m[1] == 0:
            nxt.closeHandToolLft()
        elif m[1] == 1:
            nxt.openHandToolLft()
        nxt.setJointAngles(m[2:27],tm=m[0]) # no hand open-close control
    
    main_proc_print("Finish!!")
        
        
            
        