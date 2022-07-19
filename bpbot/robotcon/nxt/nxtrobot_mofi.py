import os
import sys
import numpy as np
import argparse
from nxtrobot_client import NxtRobot

if __name__ == "__main__":
    # absolute path needed
    mfik_path =  "/home/hlab/bpbot/data/motion/motion_ik.dat"
    parser = argparse.ArgumentParser(description='ik file path')
    parser.add_argument('--filepath','-f', type=str, 
                        help='ik file path', default=mfik_path)
    args = parser.parse_args()

    motion_seq = np.loadtxt(args.filepath)
    nxt = NxtRobot(host='[::]:15005')

    print("Start robot execution .. ")
    """
    each line for mfik: 
    m[0]: motion time (1)
    m[1]: hand action(1)
        0: close
        1: open
        2: stay
        3: pause flag
    m[2:17]: whole body joint (15) in degrees 
    m[18:21]: gripper control (4)
    """

    for i, m in enumerate(motion_seq):
        if m[1] == 0:
            nxt.closeHandToolLft()
        elif m[1] == 1:
            nxt.openHandToolLft()
        nxt.setJointAngles(m[2:27],tm=m[0]) # no hand open-close control
    print("Finish!!")
        
        
            
        
