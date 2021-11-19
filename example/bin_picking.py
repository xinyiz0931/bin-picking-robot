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

from driver.phoxi import phoxi_client as pclt
from grasp.graspability import Gripper, Graspability
from motion.motion_generator import Motion
# import learning.predictor.predict_client as pdclt
from utils.base_utils import *
from utils.transform_util import *
from utils.vision_utils import *

def get_point_cloud(save_dir):
    """
    1. capture point cloud and get numpy array
    2. pose processing and convert to depth map
    3. save depth image
    return: point array, raw image, smoothed image
    """
    
    '''1. capture point cloud and get numpy array'''
    main_proc_print("Capture point cloud ... ")

    pxc = pclt.PhxClient(host="127.0.0.1:18300")
    pxc.triggerframe()
    pc = pxc.getpcd()

    # pcd = o3d.io.read_point_cloud(pc_path)
    # pc = np.asarray(pcd.points)

    '''2. pose processing and convert to depth map'''

    # test if `pc` is empty
    # if pc[:, 2].all() == 0:
    #     warning_print("Point cloud is empty ... capture again ... ")
    #     pxc.triggerframe()
    #     pc = pxc.getpcd()
        
    main_proc_print("Convert point cloud to depth map ... ")

    rotated_pc = rotate_point_cloud(pc)
    gray_array = rotated_pc[:, 2]
    img = normalize_depth_map(gray_array, max_distance, min_distance, width, height)
    
    img_blur = cv2.medianBlur(img,5)
    result_print("Depth map : shape=({}, {})".format(width, height))

    # cv2.imshow("windows", img_blur)    
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    # cv2.imwrite(os.path.join(save_dir, "depth_raw.png"), img)
    # cv2.imwrite(os.path.join(save_dir, "depth.png"), img_blur)
    # cv2.imwrite('./exp/draw/{}'.format(input_name), cimg)
    return pc

def detect_nontangle_grasp_point(gripper, n_grasp, img_path, margins, emap):
    """Detect non-tangle grasp point using graspability

    Arguments:
        img_path {str} -- image path

    Returns:
        grasps -- grasp candidates, if detections fails, return 
    """
    (top_margin,left_margin,bottom_margin,right_margin) = margins
    img = cv2.imread(img_path)
    # img = adjust_grayscale(img)
    

    # cropped the necessary region (inside the bin)
    height, width, _ = img.shape
    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = im_cut.shape
    main_proc_print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    
    im_adj = adjust_grayscale(im_cut)
    hand_open_mask, hand_close_mask = gripper.hand_model()

    # method = Graspability(rotation_step=45, depth_step=40, handdepth=30)
    method = Graspability(rotation_step=45, depth_step=40, handdepth=30)
    # generate graspability map
    main_proc_print("Generate graspability map  ... ")
    candidates = method.combined_graspability_map(
        im_adj, hand_open_mask=hand_open_mask, hand_close_mask=hand_close_mask, merge_mask=emap)

    # detect grasps
    main_proc_print("Detect grasp poses ... ")
    grasps = method.grasp_detection(
        candidates, n=n_grasp, h=cropped_height, w=cropped_width)

    if grasps != []:
        main_proc_print("Success! ... ")
        return grasps, im_adj, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, im_adj,img


def detect_grasp_point(gripper, n_grasp, img_path, margins):
    """Detect grasp point using graspability

    Arguments:
        img_path {str} -- image path

    Returns:
        grasps -- grasp candidates, if detections fails, return 
    """
    (top_margin,left_margin,bottom_margin,right_margin) = margins
    img = cv2.imread(img_path)
    # img = adjust_grayscale(img)
    

    # cropped the necessary region (inside the bin)
    height, width, _ = img.shape
    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = im_cut.shape
    main_proc_print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    
    im_adj = adjust_grayscale(im_cut)
    hand_open_mask, hand_close_mask = gripper.hand_model()

    method = Graspability(rotation_step=45, depth_step=50, handdepth=50)

    # generate graspability map
    main_proc_print("Generate graspability map  ... ")
    candidates = method.graspability_map(
        im_adj, hand_open_mask=hand_open_mask, hand_close_mask=hand_close_mask)

    # detect grasps
    main_proc_print("Detect grasp poses ... ")
    grasps = method.grasp_detection(
        candidates, n=n_grasp, h=cropped_height, w=cropped_width)

    if grasps != []:
        main_proc_print("Success! ... ")
        return grasps, im_adj, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, im_adj,img

# def draw_grasps(gripper, img_path, grasps, best_grasp, margins):
#     # draw grasps
#     # drawc, _ = gripper.draw_grasps(
#     #     grasps, img.copy(), im_cut.copy(), left_margin, top_margin, all=False)
#     # _, drawf = gripper.draw_grasps(
#     #     grasps, img.copy(), im_cut.copy(), left_margin, top_margin, all=True)
#     (top_margin,left_margin,bottom_margin,right_margin) = margins
#     img = cv2.imread(img_path)
#     # cropped the necessary region (inside the bin)

#     im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
#     drawc, drawf = gripper.draw_uniform_grasps(
#         grasps, img, im_cut, left_margin, top_margin)
#     drawc, drawf = gripper.draw_grasp(
#         best_grasp, drawf, drawc, left_margin, top_margin)
#     return drawc

# important! do not delete! 
# def draw_all_grasp(gripper, img_path, grasps, margins):
#     (top_margin,left_margin,bottom_margin,right_margin) = margins

#     img = cv2.imread(img_path)
#     # cropped the necessary region (inside the bin)

#     im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
#     drawc = gripper.draw_grasp(grasps, im_cut)
#     grasps= np.array(grasps, dtype=np.uint8)
#     grasps[:,1] += left_margin
#     grasps[:,2] += top_margin
#     drawf = gripper.draw_grasp(grasps, img)

#     return drawc, drawf

def draw_all_grasp(gripper, img_path, grasps, margins):
    (top_margin,left_margin,bottom_margin,right_margin) = margins

    img = cv2.imread(img_path)
    # cropped the necessary region (inside the bin)

    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    drawc = gripper.draw_grasp(grasps, im_cut)
    # grasps= np.array(grasps, dtype=np.uint8)
    # grasps[:,1] += left_margin
    # grasps[:,2] += top_margin
    # drawf = gripper.draw_grasp(grasps, img)

    return drawc

def convert_coordinates(grasp_point, pc, img_path, calib_path):
    """
    1. replace bad point to adjust height
    2. image (x,y) -> camera (x,y,z)
    3. camera (x,y,z) -> robot (x,y,z)
    """
    result_print("Grasp point (crop) : [{}, {}, {}]".format(grasp_point[1], grasp_point[2],grasp_point[4]))
    full_image_x = grasp_point[1] + left_margin
    full_image_y = grasp_point[2] + top_margin
    result_print("Grasp point (full) : [{}, {}, {}]".format(full_image_x, full_image_y, 45*grasp_point[4]))
    # only when height value is unnatural, execute `replace_bad_point`
    flag, (image_x, image_y) = replace_bad_point(img_path, (full_image_x, full_image_y))

    if flag: # first time adjust height
        warning_print("Seek the neighbor point to adjust height")

    offset = image_y * width + image_x
    [camera_x, camera_y, camera_z] = pc[offset]/1000 # unit: m
    result_print("To camera coordinate : [{:.3f}, {:.3f}, {:.3f}]".format(camera_x, camera_y, camera_z))
    
    x, y, z, a = camera_to_robot(
        camera_x, camera_y, camera_z, grasp_point[4], calib_path
    )
    result_print("To robot coordinate : [{:.3f}, {:.3f}, {:.3f}]".format(x, y, z))

    return x,y,z,a


def generate_motion(filepath, ee_pose, action):
    x,y,z,a = ee_pose

    ###### Object: screw ######
    # h_offset = 0.018
    # z -= h_offset 
    ###### Object: harness ######
    # h_offset = 0.013
    # z -= h_offset 
    z += 0.005
    exec_flag = 0
    # for both accuracy and safety, be careful with the robot coordinate
    # if z is too small or too large, DO NOT generate motion file, just empty it
    generator = Motion(filepath)

    # out of robot workspace
    if z < 0.011: # table
        warning_print("Z value is too small!! ")
        generator.empty_motion_generator()
        main_proc_print("Fail! Please try again ... ")
    
    elif z >= 0.130: 
        warning_print("Z value is too large!! ")
        generator.empty_motion_generator()
        main_proc_print("Fail! Please try again ... ")
    elif x <= 0.367 and y >= 0: 
        warning_print("X value is too small!! ")
        generator.empty_motion_generator()
        main_proc_print("Fail! Please try again ... ")
    elif x >= 0.644:
        warning_print("X value is too large!! ")
        generator.empty_motion_generator()
        main_proc_print("Fail! Please try again ... ")
    elif y>= 0.252 or y <= -0.243:
        warning_print("Y value is too large!! ")
        generator.empty_motion_generator()
        main_proc_print("Fail! Please try again ... ")
    elif action == -1:
        warning_print("No grasp available!  ")
        generator.empty_motion_generator()
        main_proc_print("Fail! Please try again ... ")
    else:
        exec_flag=1
        # generator.motion_generator(x,y,z,a)
        if action == 0:
            important_print("Action scheme No.{}".format(action))
            generator.motion_generator_dl(x,y,z,a)
        elif action == 1:
            important_print("Action scheme No.{}".format(action))
            generator.motion_generator_half(x,y,z,a)
        elif action == 2:
            important_print("Action scheme No.{}".format(action))
            generator.motion_generator_half_spin(x,y,z,a)
        elif action == 3:
            important_print("Action scheme No.{}".format(action))
            generator.motion_generator_full(x,y,z,a)
        elif action == 4:
            important_print("Action scheme No.{}".format(action))
            generator.motion_generator_full_spin(x,y,z,a)
        elif action == 5:
            important_print("Action scheme No.{}".format(action))
            generator.motion_generator_two_full(x,y,z,a)
        elif action == 6:
            important_print("Action scheme No.{}".format(action))
            generator.motion_generator_two_full_spin(x,y,z,a)
            # generator.motion_generator_two_full(x,y,z,a)
    return exec_flag


