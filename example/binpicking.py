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
from grasping.graspability import Graspability
from grasping.gripper import Gripper
from motion.motion_generator import Motion
# import learning.predictor.predict_client as pdclt
from utils.base_utils import *
from utils.transform_utils import *
from utils.vision_utils import *

def get_point_cloud(save_dir):
    """
    1. capture point cloud and get numpy array
    2. pose processing and convert to depth map
    return: point array, raw image, smoothed image
    """
    
    '''1. capture point cloud and get numpy array'''
    main_proc_print("Capture point cloud ... ")

    pxc = pclt.PhxClient(host="127.0.0.1:18300")
    pxc.triggerframe()
    pc = pxc.getpcd()

    '''2. pose processing and convert to depth map'''  
    main_proc_print("Convert point cloud to depth map ... ")

    rotated_pc = rotate_point_cloud(pc)
    gray_array = rotated_pc[:, 2]
    img = normalize_depth_map(gray_array, max_distance, min_distance, width, height)
    
    img_blur = cv2.medianBlur(img,5)
    result_print("Depth map : shape=({}, {})".format(width, height))

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
        important_print(f"Success! Detect {len(grasps)} grasps! ")
        return grasps, im_adj, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, im_adj,img

def detect_grasp_width_adjusted(n_grasp, img_path, margins, g_params, h_params):
    """detect grasp points with adjusting width"""
    (top_margin,left_margin,bottom_margin,right_margin) = margins
    img = cv2.imread(img_path)

    # cropped the necessary region (inside the bin)
    height, width, _ = img.shape
    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = im_cut.shape
    main_proc_print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    
    im_adj = adjust_grayscale(im_cut)

    (finger_h, finger_w, open_w, gripper_size) = h_params

    min_open_w = 25
    open_step = 20
    
    all_candidates = []

    while open_w >= min_open_w:
        # ------------------
        gripper = Gripper(finger_w=finger_w, finger_h=finger_h, open_w=open_w, gripper_size=gripper_size)
        hand_open_mask, hand_close_mask = gripper.create_hand_model()

        (rstep, dstep, hand_depth) = g_params
        method = Graspability(rotation_step=rstep, depth_step=dstep, hand_depth=hand_depth)

        # generate graspability map
        main_proc_print("Generate graspability map  ... ")
        candidates = method.width_adjusted_graspability_map(
            im_adj, hand_open_mask=hand_open_mask, hand_close_mask=hand_close_mask,width_count=open_w)
        all_candidates += candidates
        # ------------------
        open_w -= open_step

    if all_candidates != []:
    # detect grasps
        main_proc_print("Detect grasp poses ... ")
        grasps = method.grasp_detection(
            all_candidates, n=n_grasp, h=cropped_height, w=cropped_width)
        # print(grasps)
        if grasps != [] :
            important_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            drawn_input_img = gripper.draw_grasp(grasps, im_adj.copy(), (73,192,236))
            # cv2.imwrite("/home/xinyi/Pictures/g_max_pixel_area.png", drawn_input_img)
            cv2.imshow("grasps", drawn_input_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            return grasps, im_adj, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, im_adj,img

def detect_target_oriented_grasp(n_grasp, img_dir, margins, g_params, h_params):
    """Detect grasp point with target-oriented graspability algorithm"""

    (top_margin,left_margin,bottom_margin,right_margin) = margins
    
    img_path = os.path.join(img_dir, "depth.png")
    touch_path = os.path.join(img_dir, "mask_target.png")
    conflict_path = os.path.join(img_dir, "mask_others.png")

    # temporal
    GripperD = 25

    img = cv2.imread(img_path)
    depth = cv2.imread(img_path, 0)

    # conflict_mask = np.zeros(touch_mask.shape, dtype = "uint8")
    mask_target = cv2.imread(touch_path, 0)
    mask_others = cv2.imread(conflict_path, 0)
    touch_mask = cv2.bitwise_and(depth, mask_target)
    conflict_mask  = cv2.bitwise_and(depth, mask_others)

    # cropped the necessary region (inside the bin)
    height, width, _ = img.shape
    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = im_cut.shape
    main_proc_print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    # im_adj = adjust_grayscale(im_cut)
    im_adj = im_cut

    # create gripper
    (finger_h, finger_w, open_w, gripper_size) = h_params
    gripper = Gripper(finger_w=finger_w, finger_h=finger_h, open_w=open_w, gripper_size=gripper_size)
    hand_open_mask, hand_close_mask = gripper.create_hand_model()

    (rstep, dstep, hand_depth) = g_params
    method = Graspability(rotation_step=rstep, depth_step=dstep, hand_depth=hand_depth)

    # generate graspability map
    all_candidates = []
    for d in np.arange(0, 201, 50):
        _, Wt = cv2.threshold(touch_mask, d + GripperD, 255, cv2.THRESH_BINARY)
        _, Wc = cv2.threshold(depth, d, 255, cv2.THRESH_BINARY)

        # Wc = cv2.bitwise_or(Wc, cv2.subtract(touch_mask, Wt))
        main_proc_print("Generate graspability map  ... ")
        candidates = method.target_oriented_graspability_map(
            im_adj, hand_open_mask=hand_open_mask, hand_close_mask=hand_close_mask,
            Wc=Wc, Wt=Wt)

        all_candidates += candidates
    
    # detect grasps

    if all_candidates != []:
    # detect grasps
        main_proc_print("Detect grasp poses ... ")
        grasps = method.grasp_detection(
            all_candidates, n=n_grasp, h=cropped_height, w=cropped_width, _distance=50)
        # print(grasps)
        if grasps != [] :
            important_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            drawn_input_img = gripper.draw_grasp(grasps, im_adj.copy(), (73,192,236))
            # cv2.imwrite("/home/xinyi/Pictures/g_max_pixel_area.png", drawn_input_img)
            cv2.imshow("grasps", drawn_input_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            return grasps, im_adj, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, im_adj,img


def detect_grasp_point(n_grasp, img_path, margins, g_params, h_params):
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
    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = im_cut.shape
    main_proc_print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    im_adj = adjust_grayscale(im_cut)

    (finger_h, finger_w, open_w, gripper_size) = h_params
    gripper = Gripper(finger_w=finger_w, 
                      finger_h=finger_h, 
                      open_w=open_w, 
                      gripper_size=gripper_size)

    hand_open_mask, hand_close_mask = gripper.create_hand_model()

    (rstep, dstep, hand_depth) = g_params
    method = Graspability(rotation_step=rstep, 
                          depth_step=dstep, 
                          hand_depth=hand_depth)

    # generate graspability map
    main_proc_print("Generate graspability map  ... ")
    candidates = method.graspability_map(im_adj, 
                                         hand_open_mask=hand_open_mask, 
                                         hand_close_mask=hand_close_mask)
    
    if candidates != []:
    # detect grasps
        main_proc_print(f"Detect grasp poses from {len(candidates)} candidates ... ")
        grasps = method.grasp_detection(
            candidates, n=n_grasp, h=cropped_height, w=cropped_width)
        # print(grasps)
        if grasps != [] :
            important_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            drawn_input_img = gripper.draw_grasp(grasps, im_adj.copy(), (73,192,236))
            # cv2.imshow("window", drawn_input_img)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            return grasps, im_adj, drawn_input_img
        else:
            warning_print("Grasp detection failed! No grasps!")
            return None, im_adj,img

    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, im_adj,img


def transform_coordinates(grasp_point, pc, img_path, calib_path, margins):
    """
    1. replace bad point to adjust height
    2. image (x,y) -> camera (x,y,z)
    3. camera (x,y,z) -> robot (x,y,z)
    """
    (top_margin,left_margin,bottom_margin,right_margin) = margins
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


