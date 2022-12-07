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

    print("[*] Extra motion sampling")

    gname = 'rarm'

    tmlsit = motion_seq[:,0]
    print(tmlsit)
    if gname == 'torso':
        angleslist = motion_seq[:,1:2]
    elif gname == 'head':
        angleslist = motion_seq[:,2:4]
    elif gname == 'rarm':
        angleslist = motion_seq[:,4:10]
    elif gname == 'larm':
        angleslist = motion_seq[:,10:16]
    print(angleslist)
    nxt.playSmoothMotionSeq(gname, motion_seq)



    # seq = []
    # itvl = 10
    # # i : 2-3, 3-4
    # fling = motion_seq[2:5]
    # for i, jnt in enumerate(fling):
    #     if i == len(fling) - 1:
    #         jnt[0] = tm 
    #         seq.append(jnt)
    #         break
    #     jnt_next = fling[i+1]
    #     delta = (jnt_next-jnt)/itvl
    #     tm = jnt_next[0]/itvl
    #     jnt[0] = tm 
    #     seq.append(jnt)
    #     for k in range(itvl-1):
    #         jnt_ = delta*(k+1)+jnt
    #         jnt_[0] = tm
    #         seq.append(jnt_)
    # seq = np.round(np.array(seq), 6)
    # final_seq = np.vstack((motion_seq[:2], seq, motion_seq[5:]))
    # print(final_seq.shape)
    # np.savetxt("/home/hlab/bpbot/data/motion/motion_ik_uni.dat", final_seq, fmt='%.06f')

    # print("[*] Start robot execution .. ")

    # # nxt.playMotionSeq(motion_seq)
    # nxt.playMotionSeq(final_seq)
    # print("[*] Finish! ")

        
