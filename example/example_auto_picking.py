import os
import sys
import math
import random
from numpy.core.numeric import full

from numpy.lib.type_check import _imag_dispatcher

# execute the script from the root directory etc. ~/src/myrobot
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import configparser
import matplotlib.pyplot as plt
from datetime import datetime as dt

from example.bin_picking import *
from driver.phoxi import phoxi_client as pclt
from grasp.graspability import Gripper, Graspability
from motion.motion_generator import Motion
import learning.predictor.predict_client as pdclt
from utils.base_utils import *
from utils.calib_utils import *
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
    final_draw_path = os.path.join(ROOT_DIR, "vision/depth/final_result.png")


    # ======================= get config info ============================
    config = configparser.ConfigParser()
    config.read(config_path)

    width = int(config['GRASP']['width'])
    height = int(config['GRASP']['height'])

    left_margin = int(config['GRASP']['left_margin'])
    top_margin = int(config['GRASP']['top_margin'])
    right_margin = int(config['GRASP']['right_margin'])
    bottom_margin = int(config['GRASP']['bottom_margin'])
    max_distance = float(config['GRASP']['max_distance'])
    min_distance = float(config['GRASP']['min_distance'])
    dismiss_area_width = float(config['GRASP']['dismiss_area_width'])
    hand_width = float(config['GRASP']['hand_width'])

    # ======================== get depth img =============================
    # point_array = get_point_cloud(depth_dir) 

    # =======================  compute grasp =============================
    # prepare hand model
    gripper = Gripper()
    grasps, input_img, full_image = detect_grasp_point(gripper,img_path, (top_margin,left_margin,bottom_margin,right_margin))
    cv2.imwrite(crop_path, input_img) # save first!!
    # best_grasp = grasps[0]

    # =======================  picking policy ===========================
    # from learning.predictor import grasp_policy, predict_patch
    # model_dir = "/home/xinyi/Workspace/myrobot/learning/model/Logi_AL_20210827_145223.h5"
    # from tensorflow.keras.models import load_model
    # model = load_model(model_dir)    
    # model.compile(optimizer='adam',
    #                 loss='binary_crossentropy',
    #                 metrics='accuracy')


    if grasps is None:
        best_action = -1
        rx,ry,rz,ra = np.zeros(4)
    else:
        pdc = pdclt.PredictorClient()
        grasps2bytes=np.ndarray.tobytes(np.array(grasps))
        predict_result= pdc.predict(imgpath=crop_path, grasps=grasps2bytes)

        best_action = predict_result.action
        best_grasp = grasps[predict_result.graspno]


        # best_grasp , best_action = predict_patch(model, input_img, grasps)

        best_grasp = grasps[0]

        # best_action = random.sample(list(range(6)),1)[0] # random policy

        
        # best_action = 6 # all complex policy
        best_action = 0 # graspability policy

        # best_grasp, best_action = grasp_policy(model,input_img, grasps)

        # time_print("direct policy: {:.2f}s".format(end-start))

        cimg = draw_grasps(gripper,img_path, grasps, best_grasp, (top_margin,left_margin,bottom_margin,right_margin))
        cv2.imwrite(final_draw_path, cimg)
        rx,ry,rz,ra = convert_coordinates(best_grasp, point_array, img_path, calib_path)

    # best_action, best_prob = predict(input_img, best_grasp[1], best_grasp[2])

    # =======================  generate motion ===========================
    
    flag=generate_motion(mf_path, [rx,ry,rz,ra], best_action) 
    end = timeit.default_timer()
  
    main_proc_print("Total: {:.2f}s".format(end-start))
   

    cv2.imwrite(os.path.join(depth_dir, "depthc.png"), input_img)
    # print("Time cost: ", end-start)

    # # ======================= Record the data ============================
    if flag:
        main_proc_print("Save the results! ")
        tdatetime = dt.now()
        tstr = tdatetime.strftime('%Y%m%d%H%M%S')
        input_name = "{}_{}_{}_{}_.png".format(tstr,best_grasp[1],best_grasp[2],best_action)
        cv2.imwrite('./exp/{}'.format(input_name), input_img)
        cv2.imwrite('./exp/draw/{}'.format(input_name), cimg)
        cv2.imwrite('./exp/full/{}'.format(input_name), full_image)


if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
