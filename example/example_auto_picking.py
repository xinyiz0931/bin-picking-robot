import os
import sys
import math
import random
# execute the script from the root directory etc. ~/src/myrobot
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import configparser
import matplotlib.pyplot as plt
from datetime import datetime as dt

from example.binpicking import *
from driver.phoxi import phoxi_client as pclt
from grasping.graspability import Graspability
from grasping.gripper import Gripper
from motion.motion_generator import Motion
import learning.predictor.predict_client as pdclt
from utils.base_utils import *
from utils.transform_utils import *
from utils.vision_utils import *

def main():
    main_proc_print("Start! ")
    # ========================== define path =============================
    ROOT_DIR = os.path.abspath("./")
    calib_dir = os.path.join(ROOT_DIR, "vision/calibration/")
    depth_dir = os.path.join(ROOT_DIR, "vision/depth/")

    # pc_path = os.path.join(ROOT_DIR, "vision/pointcloud/out.ply")
    img_path = os.path.join(ROOT_DIR, "vision/depth/depth.png")
    crop_path = os.path.join(ROOT_DIR, "vision/depth/depthc.png")
    config_path = os.path.join(ROOT_DIR, "cfg/config.ini")
    calib_path = os.path.join(ROOT_DIR, "vision/calibration/calibmat.txt")
    mf_path = os.path.join(ROOT_DIR, "motion/motion.dat")
    draw_path = os.path.join(ROOT_DIR, "vision/depth/final_result.png")


    # ======================= get config info ============================
    config = configparser.ConfigParser()
    config.read(config_path)

    # image / point cloud 
    width = int(config['IMAGE']['width'])
    height = int(config['IMAGE']['height'])
    left_margin = int(config['IMAGE']['left_margin'])
    top_margin = int(config['IMAGE']['top_margin'])
    right_margin = int(config['IMAGE']['right_margin'])
    bottom_margin = int(config['IMAGE']['bottom_margin'])
    max_distance = float(config['IMAGE']['max_distance'])
    min_distance = float(config['IMAGE']['min_distance'])

    finger_w = float(config['GRASP']['finger_width'])
    finger_h = float(config['GRASP']['finger_height'])
    open_w = float(config['GRASP']['gripper_width'])
    hand_size = int(config['GRASP']['hand_template_size'])

    rotation_step = float(config['GRASP']['rotation_step'])
    depth_step = float(config['GRASP']['rotation_step'])
    hand_depth = float(config['GRASP']['hand_depth'])
    dismiss_area_width = float(config['GRASP']['dismiss_area_width'])

    exp_mode = int(config['EXP']['exp_mode'])
    

    # ======================== get depth img =============================
    # point_array = get_point_cloud(depth_dir) 

    # =======================  compute grasp =============================
    # prepare all kinds of parameters
    margins =  (top_margin,left_margin,bottom_margin,right_margin)
    g_params = (rotation_step, depth_step, hand_depth)
    h_params = (finger_h, finger_w, open_w, hand_size)

    grasps, input_img, output_img = detect_grasp_point(n_grasp=10, 
                                                       img_path=img_path, 
                                                       margins=margins, 
                                                       g_params=g_params, 
                                                       h_params=h_params)

    # =======================  picking policy ===========================
    if grasps is None:
        best_action = -1
        rx,ry,rz,ra = np.zeros(4)
    else:
        # four policies
        if exp_mode == 0:
            # 0 -> graspaiblity
            best_grasp = grasps[0]
            best_action = 0 

        elif exp_mode == 1: 
            # 1 -> proposed circuclar picking
            pdc = pdclt.PredictorClient()
            grasps2bytes=np.ndarray.tobytes(np.array(grasps))
            predict_result= pdc.predict(imgpath=crop_path, grasps=grasps2bytes)
            best_action = predict_result.action
            best_grasp = grasps[predict_result.graspno]

        elif exp_mode == 2:
            # 2 -> random circular picking
            best_grasp = grasps[0]
            best_action = random.sample(list(range(6)),1)[0]
       
        rx,ry,rz,ra = transform_coordinates(best_grasp, point_array, img_path, calib_path, margins)
        
    # # =======================  generate motion ===========================
    # success_flag = generate_motion(mf_path, [rx,ry,rz,ra], best_action) 
  
    # ======================= Record the data ===================s=========
    main_proc_print("Save the results! ")
    cv2.imwrite(crop_path, input_img)
    cv2.imwrite(draw_path, output_img)

    # if success_flag:
    #     tdatetime = dt.now()
    #     tstr = tdatetime.strftime('%Y%m%d%H%M%S')
    #     input_name = "{}_{}_{}_{}_.png".format(tstr,best_grasp[1],best_grasp[2],best_action)
    #     cv2.imwrite('./exp/{}'.format(input_name), input_img)
    #     cv2.imwrite('./exp/draw/{}'.format(input_name), cimg)
    #     cv2.imwrite('./exp/full/{}'.format(input_name), full_image)


if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
