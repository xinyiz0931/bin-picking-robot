"""
A python scripts to for bin picking functions
Arthor: Xinyi
Date: 2020/5/11
"""
import os
import sys
# execute the script from the root directory etc. ~/src/myrobot
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from myrobot.grasping import Graspability, Gripper
from myrobot.motion import Motion
from myrobot.utils import *

def get_point_cloud(save_dir, max_distance, min_distance, width, height):
    """
    1. Capture point cloud and get numpy array
    2. Pose refinement and convert to depth map
    3. Save images (raw, smoothed)
    Return: 
        pc {array} -- (point num x 3)
    """
    # 1. ===================================================
    main_proc_print("Capture point cloud ... ")
    from myrobot.driver import phoxi_client as pclt
    pxc = pclt.PhxClient(host="127.0.0.1:18300")
    pxc.triggerframe()
    pc = pxc.getpcd()

    # 2. ===================================================
    main_proc_print("Convert point cloud to depth map ... ")
    rotated_pc = rotate_point_cloud(pc)
    gray_array = rotated_pc[:, 2]

    # 3. ===================================================
    img = normalize_depth_map(gray_array, max_distance, min_distance, width, height)
    img_blur = cv2.medianBlur(img,5)
    cv2.imwrite(os.path.join(save_dir, "depth_raw.png"), img)
    cv2.imwrite(os.path.join(save_dir, "depth.png"), img_blur)
    result_print("Depth map : shape=({}, {})".format(width, height))

    return pc

def crop_roi(img_path, margins=None):
    """
    Read image crop roi
    """
    img = cv2.imread(img_path)
    height, width, _ = img.shape
    if margins is None:
        (top_margin,left_margin,bottom_margin,right_margin) = (0,0,height,width)
    else:
        (top_margin,left_margin,bottom_margin,right_margin) = margins

    # cropped the necessary region (inside the bin)
    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    return im_cut

def draw_grasps(grasps, img_path, h_params, top_idx=0, color=(73,192,236), top_color=(0,255,0)):
    """
    Read image and draw grasps
    """
    img = cv2.imread(img_path)
    (finger_w, finger_h, open_w, gripper_size) = h_params
    gripper = Gripper(finger_w, finger_h, open_w, gripper_size)
    draw_img = gripper.draw_grasp(grasps, img.copy(), top_idx, color, top_color)
    return draw_img

def detect_edge(img_path, t_params, margins=None):
    """
    Read image and detect edge
    """
    from myrobot.tangle_solution import LineDetection
    if margins is not None:
        img = crop_roi(img_path, margins)
    else:
        img = cv2.imread(img_path)

    (length_thre, distance_thre, sliding_size, sliding_stride, c_size) = t_params
    
    norm_img = cv2.resize(adjust_grayscale(img), (c_size,c_size))

    ld = LineDetection(length_thre=length_thre,distance_thre=distance_thre)
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)
    return drawn, lines_num

def get_entanglement_map(img_path, t_params, margins=None):
    """
    Read image and generate entanglement map
    """
    from myrobot.tangle_solution import LineDetection, EntanglementMap
    if margins is not None:
        img = crop_roi(img_path, margins)
    else:
        img = cv2.imread(img_path)

    (length_thre, distance_thre, sliding_size, sliding_stride, c_size) = t_params
    
    norm_img = cv2.resize(adjust_grayscale(img), (c_size,c_size))

    ld = LineDetection(length_thre=length_thre,distance_thre=distance_thre)
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)
    
    em = EntanglementMap(length_thre, distance_thre, sliding_size, sliding_stride)
    emap, wmat_vis,w,d = em.entanglement_map(norm_img)
    lmap = em.line_map(norm_img)
    bmap = em.brightness_map(norm_img)
    return img, emap


def detect_grasp_point(n_grasp, img_path, g_params, h_params, margins=None):
    """Detect grasp point using graspability
    Parameters:
        n_grasp {int} -- number of grasps you want to output
        img_path {str} -- image path
        g_params {tuple} -- graspaiblity parameters
        h_params {tuple} -- hand (gripper) parameters
        margins {tuple} -- crop roi if you need
    Returns:
        grasps {list} -- grasp candidates [g,x,y,z,a,rot_step, depth_step]
        img {array} -- cropped input image
        drawn {array} -- image that draws detected grasps
    """
    if margins is not None:
        img = crop_roi(img_path, margins)
    else:
        img = cv2.imread(img_path)

    img_adj = adjust_grayscale(img)
    cropped_height, cropped_width, _ = img.shape
    (finger_w, finger_h, open_w, gripper_size) = h_params
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
    candidates = method.graspability_map(img_adj, 
                                         hand_open_mask=hand_open_mask, 
                                         hand_close_mask=hand_close_mask)
    
    if candidates != []:
    # detect grasps
        main_proc_print(f"Detect grasp poses from {len(candidates)} candidates ... ")
        grasps = method.grasp_detection(
            candidates, n=n_grasp, h=cropped_height, w=cropped_width)
        if grasps != [] :
            important_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            drawn = gripper.draw_grasp(grasps, img_adj.copy(), top_idx=0) 
            #cv2.imshow("window", drawn)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            return grasps, img_adj, drawn
        else:
            warning_print("Grasp detection failed! No grasps!")
            return None, img, img

    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, img, img

def detect_nontangle_grasp_point(n_grasp, img_path, g_params, h_params, t_params, margins=None):
    """Detect grasp point using graspability
    Parameters:
        n_grasp {int} -- number of grasps you want to output
        img_path {str} -- image path
        g_params {tuple} -- graspaiblity parameters
        h_params {tuple} -- hand (gripper) parameters
        t_params {tuple} -- entanglemet map parameters
        margins {tuple} -- crop roi if you need
    Returns:
        grasps {list} -- grasp candidates [g,x,y,z,a,rot_step, depth_step]
        img {array} -- cropped input image
        drawn {array} -- image that draws detected grasps
    """
    if margins is not None:
        img, emap = get_entanglement_map(img_path, t_params, margins)
    else:
        img, emap = get_entanglement_map(img_path, t_params)

    (finger_w, finger_h, open_w, gripper_size) = h_params
    gripper = Gripper(finger_w=finger_w, 
                      finger_h=finger_h, 
                      open_w=open_w, 
                      gripper_size=gripper_size)

    (rstep, dstep, hand_depth) = g_params
    method = Graspability(rotation_step=rstep, 
                          depth_step=dstep, 
                          hand_depth=hand_depth)
    
    cropped_height, cropped_width, _ = img.shape
    hand_open_mask, hand_close_mask = gripper.create_hand_model()

    main_proc_print("Generate graspability map  ... ")
    candidates = method.combined_graspability_map(img, hand_open_mask, hand_close_mask, emap)
    
    if candidates != []:
        # detect grasps
        main_proc_print(f"Detect grasp poses from {len(candidates)} candidates ... ")
        grasps = method.grasp_detection(
            candidates, n=n_grasp, h=cropped_height, w=cropped_width)

        if grasps != [] :
            important_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            drawn = gripper.draw_grasp(grasps, img.copy(), top_idx=0) 
            # cv2.imshow("window", drawn)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            return grasps, img, drawn
        else:
            warning_print("Grasp detection failed! No grasps!")
            return None, img, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, img, img

def predict_action_grasp(grasps, crop_path):
    """
    Returns:
        best_action {int} -- index of best action within range(0,1,...,6)
        best_graspno {int} -- index of best grasp among grasps
    """
    from myrobot.prediction import predict_client as pdclt
    pdc = pdclt.PredictorClient()
    grasps2bytes=np.ndarray.tobytes(np.array(grasps))
    predict_result= pdc.predict(imgpath=crop_path, grasps=grasps2bytes)
    best_action = predict_result.action
    best_graspno = predict_result.graspno
    return best_action, best_graspno

##################### TODO: test following code ##################
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
    im_adj = im_cut
    (finger_w, finger_h, open_w, gripper_size) = h_params

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
    (finger_w, finger_h, open_w, gripper_size) = h_params
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



def transform_coordinates(grasp_point, point_cloud, img_path, calib_path, width, margins):
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
    [camera_x, camera_y, camera_z] = point_cloud[offset]/1000 # unit: m
    result_print("To camera coordinate : [{:.3f}, {:.3f}, {:.3f}]".format(camera_x, camera_y, camera_z))
    
    x, y, z, a = camera_to_robot(
        camera_x, camera_y, camera_z, grasp_point[4], calib_path
    )
    
    z += 0.016
    if z < plane_distance:
        z = plane_distance
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
    exec_flag = 0
    # for both accuracy and safety, be careful with the robot coordinate
    # if z is too small or too large, DO NOT generate motion file, just empty it
    generator = Motion(filepath)

    if z < 0.030 or z >=0.130 or x <= 0.30 or x >= 0.644 or y >=0.252 or y <= -0.243: #table
        warning_print("Out of robot workspace!! ")
        generator.empty_motion_generator()
        main_proc_print("Fail! Please try again ... ")

    else:
        exec_flag=1
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
        else:
            warning_print("No grasp available!! ")
            generator.empty_motion_generator()
            main_proc_print("Fail! Please try again ... ")
    return exec_flag


