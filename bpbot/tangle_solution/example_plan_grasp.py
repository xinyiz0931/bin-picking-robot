import os
import sys
import math
import random
# execute the script from the root directory etc. ~/src/bpbot
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt

from bpbot.grasping import Gripper, Graspability
from bpbot.utils import *

def detect_tangle_grasp(gripper, n_grasp, img_path, margins, g_params):
    """Detect grasp point using graspability

    Arguments:
        img_path {str} -- image path

    Returns:
        grasps -- grasp candidates, if detections fails, return 
    """
    (top_margin,left_margin,bottom_margin,right_margin) = margins
    img = cv2.imread(img_path)

    # cropped the necessary region (inside the bin)
    height, width, _ = img.shape
    img_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = img_cut.shape
    main_proc_print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    
    im_adj = adjust_grayscale(img_cut)
    # im_adj = img_cut
    hand_open_mask, hand_close_mask = gripper.hand_model()

    (rstep, dstep, hand_depth, Wc, Wt) = g_params
    method = Graspability(rotation_step=rstep, depth_step=dstep, hand_depth=hand_depth)

    # generate graspability map
    main_proc_print("Generate graspability map  ... ")
    candidates = method.target_oriented_graspability_map(
        im_adj, hand_open_mask=hand_open_mask, hand_close_mask=hand_close_mask, Wc=Wc, Wt=Wt)

    # detect grasps
    main_proc_print("Detect grasp poses ... ")
    grasps = method.grasp_detection(
        candidates, n=n_grasp, h=cropped_height, w=cropped_width, _dismiss=0, _distance=0)

    if grasps != []:
        notice_print(f"Success! Detect {len(grasps)} grasps! ")
        return grasps, im_adj, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, im_adj,img

if __name__ == '__main__':

    """Configurations defined by users"""
    root_dir = os.path.abspath("./")
    img_path = os.path.join(root_dir, "data/test/depth4.png")

    # conflict = np.zeros((500,500))
    # for i in range(7):
    #     if i!=3:
    #         mask_path = os.path.join(root_dir, f"mask_{i}.png")
    #         mask = cv2.imread(mask_path, 0)
    #         conflict+=mask

    # cv2.imwrite(os.path.join(root_dir, "conflict.png"), conflict)


    # prepare hand model
    gripper = Gripper(finger_w=5, finger_h=5, gripper_width=46)
    # margins = (top_margin,left_margin,bottom_margin,right_margin)
    Wc = cv2.imread(os.path.join(root_dir, "mask_3.png"),0)
    Wt = cv2.imread(os.path.join(root_dir, "mask.png"),0)
    margins = (0,0,500,500)
    g_params = (10, 50, 50, Wc, Wt)
    grasps, input_img, full_image = detect_tangle_grasp(gripper=gripper, n_grasp=1, img_path=img_path, margins=margins, g_params=g_params)


    drawn_input_img = gripper.draw_grasp(grasps, input_img)

    # cv2.imwrite(os.path.join(root_dir, "best.png"), drawn_input_img)
    # gmap = cv2.imread("./vision\\tmp\\G_11.png", 0)
    
    plt.imshow(cv2.imread(img_path, 0), cmap='gray')
    plt.imshow(gmap, cmap='jet', alpha=0.4)
    plt.show()