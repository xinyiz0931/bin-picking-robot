import os
import sys
import math
import random
from datetime import datetime as dt

from myrobot.binpicking import *

def main():
    main_proc_print("Start! ")
    # ========================== define path =============================
    ROOT_DIR = os.path.abspath("./")
    calib_dir = os.path.join(ROOT_DIR, "data/calibration/")
    depth_dir = os.path.join(ROOT_DIR, "data/depth/")

    # pc_path = os.path.join(ROOT_DIR, "data/pointcloud/out.ply")
    img_path = os.path.join(ROOT_DIR, "data/depth/depth.png")
    crop_path = os.path.join(ROOT_DIR, "data/depth/depth_cropped.png")
    config_path = os.path.join(ROOT_DIR, "cfg/config.ini")
    calib_path = os.path.join(ROOT_DIR, "data/calibration/calibmat.txt")
    mf_path = os.path.join(ROOT_DIR, "data/motion/motion.dat")
    draw_path = os.path.join(ROOT_DIR, "data/depth/final_result.png")

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
    point_array = get_point_cloud(depth_dir, max_distance, min_distance, width, height)
    
    # ======================== crop depth image =============================
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    img_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cv2.imwrite(crop_path, img_cut)

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
