"""
A python scripts to for bin picking functions
Arthor: Xinyi
Date: 2020/5/11
"""
from logging import warning
import os
import sys
import random
import cv2
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from bpbot.grasping import Graspability, Gripper
from bpbot.motion import Motion
from bpbot.utils import *

def capture_pc():
    main_proc_print("Capture point cloud ... ")
    import bpbot.driver.phoxi.phoxi_client as pclt
    pxc = pclt.PhxClient(host="127.0.0.1:18300")
    pxc.triggerframe()
    pc = pxc.getpcd()
    gs = pxc.getgrayscaleimg()
    return pc.copy()

def pc2depth(pc, distance, width, height):
    rotated_pc = rotate_point_cloud(pc)
    gray_array = rotated_pc[:, 2] 
    # main_proc_print("Convert point cloud to depth map ...")
    max_distance, min_distance = distance["max"], distance['min']
    img = normalize_depth_map(gray_array, max_distance, min_distance, width, height)
    img_blur= cv2.medianBlur(img, 5)
    return img, img_blur

def get_point_cloud(save_dir, distance, width, height):
    """
    1. Capture point cloud and get numpy array
    2. Pose refinement and convert to depth map
    3. Save images (raw, smoothed)
    Return: 
        pc {array} -- (point num x 3)
    """
    # 1. ===================================================
    main_proc_print("Capture! ")
    import bpbot.driver.phoxi.phoxi_client as pclt
    # from bpbot.driver import phoxi_client as pclt
    pxc = pclt.PhxClient(host="127.0.0.1:18300")
    pxc.triggerframe()
    pc = pxc.getpcd()
    gs = pxc.getgrayscaleimg()
    if pc is None: return

    # 2. ===================================================
    main_proc_print("Convert point cloud to depth map ... ")
    rotated_pc = rotate_point_cloud(pc)
    gray_array = rotated_pc[:, 2]

    # 3. ===================================================
    max_distance, min_distance = distance["max"], distance['min']
    img = normalize_depth_map(gray_array, max_distance, min_distance, width, height)
    img_blur = cv2.medianBlur(img,5)
    cv2.imwrite(os.path.join(save_dir, "depth_raw.png"), img)
    cv2.imwrite(os.path.join(save_dir, "depth.png"), img_blur)
    cv2.imwrite(os.path.join(save_dir, "texture.png"), gs)

    return pc.copy()

def crop_roi(img_path, margins=None):
    """
    Read image crop roi
    """

    img = cv2.imread(img_path)
    height, width, _ = img.shape
    if margins is None:
        (top_margin,left_margin,bottom_margin,right_margin) = (0,0,height,width)
    else:
        top_margin = margins["top"]
        left_margin = margins["left"]
        bottom_margin = margins["bottom"]
        right_margin = margins["right"]
    
    # cropped the necessary region (inside the bin)
    im_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    return im_cut

def align_depth(cfg, dist=10):
    """unit: cm"""
    depth = dist * 255 / (cfg["pick"]["distance"]["max"] - cfg["pick"]["distance"]["min"])

def draw_grasps(grasps, img_path, h_params=None, top_idx=0, color=(255,0,0), top_color=(0,255,0), top_only=False):
    """
    Read image and draw grasps
    grasps = [[g,x,y,d,a,_,_], ...], or [[x,y,a], ...]
    """
    img = cv2.imread(img_path)
    finger_w = h_params["finger_width"]
    finger_h = h_params["finger_height"]
    open_w = h_params["open_width"]
    # open_w = 60
    template_size = h_params["template_size"]
    gripper = Gripper(finger_w, finger_h, open_w)
    draw_img = gripper.draw_grasp(grasps, img.copy(), top_color=top_color, top_idx=top_idx, top_only=top_only)
    return draw_img

def detect_edge(img_path, t_params, margins=None):
    """
    Read image and detect edge
    """
    from bpbot.tangle_solution import LineDetection
    if margins is not None:
        img = crop_roi(img_path, margins)
    else:
        img = cv2.imread(img_path)
    length_thre = int(t_params["length_thre"])
    distance_thre = int(t_params["distance_thre"])
    sliding_size = t_params["sliding_size"]
    sliding_stride = t_params["sliding_stride"]
    c_size = int(t_params["compressed_size"])
    # (length_thre, distance_thre, sliding_size, sliding_stride, c_size) = t_params
    
    norm_img = cv2.resize(adjust_grayscale(img), (c_size,c_size))

    ld = LineDetection(length_thre=length_thre,distance_thre=distance_thre)
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)
    return drawn, lines_num

def get_entanglement_map(img_path, t_params, margins=None):
    """
    Read image and generate entanglement map
    """
    from bpbot.tangle_solution import LineDetection, EntanglementMap
    if margins is not None:
        img = crop_roi(img_path, margins)
    else:
        img = cv2.imread(img_path)

    # (length_thre, distance_thre, sliding_size, sliding_stride, c_size) = t_params
    length_thre = int(t_params["length_thre"])
    distance_thre = int(t_params["distance_thre"])
    sliding_size = t_params["sliding_size"]
    sliding_stride = t_params["sliding_stride"]
    c_size = int(t_params["compressed_size"]) 
    norm_img = cv2.resize(adjust_grayscale(img), (c_size,c_size))

    ld = LineDetection(length_thre=length_thre,distance_thre=distance_thre)
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)
    
    em = EntanglementMap(length_thre, distance_thre, sliding_size, sliding_stride)
    emap, wmat_vis,w,d = em.entanglement_map(norm_img)
    lmap = em.line_map(norm_img)
    bmap = em.brightness_map(norm_img)
    return img, emap

def pick_or_sep(img_path, h_params, bin="pick"):
    main_proc_print(f"Infer {bin} zone using PickNet / SepNet! ")
    img = cv2.imread(img_path)
    # img = adjust_grayscale(img)
    cropped_height, cropped_width, _ = img.shape

    finger_w = h_params["finger_width"]
    finger_h = h_params["finger_height"]
    open_w = h_params["open_width"] + random.randint(-1, 5)
    template_size = h_params["template_size"]

    gripper = Gripper(finger_w=finger_w, 
                      finger_h=finger_h, 
                      open_w=open_w, 
                      gripper_size=template_size)
    
    from bpbot.module_picksep import PickSepClient
    psc = PickSepClient()
    # plt.imshow(img), plt.show() 
    # pickorsep, action = psc.infer_picknet_sepnet(imgpath=crop_path)
    if bin == "pick": 
        
        # pickorsep_pz, action_pz = psc.infer_picknet(imgpath=img_path)
        pickorsep_pz, action_pz = psc.infer_picknet(imgpath=img_path)
        warning_print(f"Pick zone scores: {action_pz[2]:.3f}, {action_pz[3]:.3f}")
        pick_x = int(action_pz[0] * cropped_width / 512)
        pick_y = int(action_pz[1] * cropped_height / 512)
        # img_adj = img
        angle_degree = gripper.point_oriented_grasp(img, [pick_x, pick_y])
        grasp_pz = [pick_x, pick_y, angle_degree*math.pi/180]
        if action_pz[2] >= 0.3 and action_pz[2]>action_pz[3]: return 0, grasp_pz
        # if (action_pz[2]-action_pz[3]) >= 0.2 and action_pz[2] > 0.3 : return 0, grasp_pz
        # if action_pz[2] > 0.3 : return 0, grasp_pz
        else: return 1, grasp_pz 
        return pickorsep_pz, grasp_pz

    elif bin == "mid": 
        pickorsep_mz, action_mz = psc.infer_picknet_sepnet(imgpath=img_path)
        # pickorsep_mz, action_mz = psc.infer_picknet(imgpath=img_path)
        
        # if mid zone has objects? sr -> 6000 
        # if (img > 40).sum() < 3000: return None 
        print((img>20).sum())
        if (img > 20).sum() < 2000: return None 
        if pickorsep_mz == 0:
            warning_print(f"Mid zone scores: {action_mz[2]:.3f}, {action_mz[3]:.3f}")
            # if action_mz[2] < 0.35 and action_mz[3] < 0.35: return None
 
            pick_x = int(action_mz[0] * cropped_width / 512)
            pick_y = int(action_mz[1] * cropped_height / 512)
            # img_adj = img
            angle_degree = gripper.point_oriented_grasp(img, [pick_x, pick_y])
            grasp_mz = [pick_x, pick_y, angle_degree*math.pi/180]
            return pickorsep_mz, grasp_mz
        else: 
            warning_print(f"Mid zone scores: {action_mz[6]:.3f}, {action_mz[7]:.3f}")

            # if action_mz[6] < 0.35 and action_mz[7] < 0.35: return None
            pull_x = int(action_mz[0] * cropped_width / 512)
            pull_y = int(action_mz[1] * cropped_height / 512)
            hold_x = int(action_mz[2] * cropped_width / 512)
            hold_y = int(action_mz[3] * cropped_height / 512)
            angle_degree = gripper.point_oriented_grasp(img, [pull_x, pull_y])
            grasp_mz = [pull_x, pull_y, angle_degree*math.pi/180, action_mz[4], action_mz[5]] 
            return pickorsep_mz, grasp_mz 

def gen_motion_pickorsep(file_path, ee_pose, v=None, motion_type="pick", dest="goal"):
    # input grasp angle a is in degree
    x,y,z,a = ee_pose
    
    if z < 0.020: 
        warning_print("Touch the plane! ")

    exec_flag = False
    generator = Motion(filepath=file_path)
    if not check_reachability((x,y,z), min_z=0.030, max_z=0.16):
        warning_print("Out of robot workspace!! ")
        generator.empty_motion_generator()
        warning_print("Fail! Please try again ... ")
    elif motion_type == "pick":
        exec_flag = True
        generator.gen_pickandplace_motion(x,y,z,a, dest=dest)
    elif motion_type == "sep" and v is not None: 
        vx,vy = v
        exec_flag = True
        generator.gen_separation_motion(x,y,z,a, vx,vy)
    return exec_flag

def gen_motion_tilt(mf_path, g_hold, g_pull, v_pull, v_len):
    generator = Motion(filepath=mf_path)
    generator.gen_separation_motion_dualarm(g_hold, g_pull, v_pull, len=v_len)
    # rad = degree * math.pi / 180
    # print("y: ", ry, " => ", ry + length* (1-math.cos(rad)))
    # print("z: ", rz, " => ", rz - length * math.sin(rad))
    
def detect_grasp_point(n_grasp, img_path, g_params, h_params, margins=None):
    """Detect grasp point using graspability
    Parameters:
        n_grasp {int} -- number of grasps you want to output
        img_path {str} -- image path
        params {dict} -- new! 
        # g_params {tuple} -- graspaiblity parameters
        # h_params {tuple} -- hand (gripper) parameters
        # margins {tuple} -- crop roi if you need
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
    # img_adj = img
    cropped_height, cropped_width, _ = img.shape

    # (finger_w, finger_h, open_w, gripper_size) = h_params
    finger_w = h_params["finger_width"]
    finger_h = h_params["finger_height"]
    # open_w = h_params["open_width"] + random.rand'int(0, 5)
    open_w = h_params["open_width"]

    gripper = Gripper(finger_w=finger_w, 
                      finger_h=finger_h, 
                      open_w=open_w)

    # drawn = gripper.draw_grasp(grasps, img_adj.copy(), top_idx=0) 
    hand_open_mask, hand_close_mask = gripper.create_hand_model()
    # (rstep, dstep, hand_depth) = g_params
    rstep = g_params["rotation_step"]
    dstep = g_params["depth_step"]
    hand_depth = g_params["hand_depth"]
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
            notice_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            # drawn = gripper.draw_grasp(grasps, img_adj.copy(), top_idx=0) 
            # drawn = gripper.draw_grasp(grasps, img_adj.copy(), top_idx=0) 
            #cv2.imshow("window", drawn)
            #cv2.waitKey()
            #cv2.destroyAllWindows()
            return grasps, img_adj
        else:
            warning_print("Grasp detection failed! No grasps!")
            return None, img

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
    rstep = g_params["rotation_step"]
    dstep = g_params["depth_step"] + random.randint(0, 5)
    hand_depth = g_params["hand_depth"]
    
    finger_w = h_params["finger_width"]
    finger_h = h_params["finger_height"]
    open_w = h_params["open_width"]
    gripper_size = h_params["template_size"]


    # (finger_w, finger_h, open_w, gripper_size) = h_params
    gripper = Gripper(finger_w=finger_w, 
                      finger_h=finger_h, 
                      open_w=open_w, 
                      gripper_size=gripper_size)

    # (rstep, dstep, hand_depth) = g_params
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
            notice_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            # drawn = gripper.draw_grasp(grasps, img.copy(), top_idx=0) 
            # cv2.imshow("window", drawn)
            # cv2.waitKey()
            # cv2.destroyAllWindows()
            return grasps, img
        else:
            warning_print("Grasp detection failed! No grasps!")
            return None, img
    else:
        warning_print("Grasp detection failed! No grasps!")
        return None, img

def predict_action_grasp(grasps, crop_path):
    """
    Parameters: 
        grasps: numpy array [[x1,y1],[x2,y2]]
    Returns:
        best_action {int} -- index of best action within range(0,1,...,6)
        best_graspno {int} -- index of best grasp among grasps
    """
    from bpbot.module_asp import asp_client as aspclt
    aspc = aspclt.ASPClient()
    grasps2bytes=np.ndarray.tobytes(np.array(grasps))
    predict_result= aspc.predict(imgpath=crop_path, grasps=grasps2bytes)
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
            notice_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
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
            notice_print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
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

def check_collision(p, v, cfg, point_cloud, margin="mid"):
    m = cfg[margin]["margin"]
    margin_points = [[m["left"], m["top"]], [m["right"], m["top"]], 
                     [m["right"], m["bottom"]], [m["left"], m["bottom"]]]
    margin_points_robot = []
    for p_i in margin_points:
        p_r = transform_image_to_robot(p_i, point_cloud, cfg)
        margin_points_robot.append(p_r)
    for i in range(4):
        m1 = margin_points_robot[i]
        m2 = margin_points_robot[(i+1)%4]
        itsct = calc_intersection(p, p+p*v, m1[:2], m2[:2])
        if is_between(m1[:2], m2[:2], itsct) and np.dot(v, itsct - p) > 0: 
            return calc_2points_distance(p, itsct)
    return None
    
# def check_collision(p, v, margin, width, calibmat_path, point_cloud):
#     margin_points = [[margin["left"], margin["top"]], [margin["right"], margin["top"]], 
#                      [margin["right"], margin["bottom"]], [margin["left"], margin["bottom"]]]
#     margin_points_robot = transform_image_to_robot(margin_points, width, calibmat_path, point_cloud)
#     orders = ["top", "right", "bottom", "left"]
#     for i in range(4):
#         m1 = margin_points_robot[i]
#         m2 = margin_points_robot[(i+1)%4]
#         itsct = calc_intersection(p, p+p*v, m1[:2], m2[:2])
#         if is_between(m1[:2], m2[:2], itsct) and np.dot(v, itsct - p) > 0: 
#             return calc_2points_distance(p, itsct)
#     return None

def transform_camera_to_robot(camera_locs, calibmat_path):
    """
    Transform camera loc to robot loc
    Use 4x4 calibration matrix
    Parameters:
        camera_loc {tuple} -- (cx,cy,cy) at camera coordinate, unit: m
        calib_path {str} -- calibration matrix file path
    Returns: 
        robot_loc {tuple} -- (rx,ry,rz) at robot coordinate, unit: m
    """
    calibmat = np.loadtxt(calibmat_path)
    robot_locs = []
    for loc in camera_locs:
        # get calibration matrix 4x4
        (cx, cy, cz) = loc
        camera_pos = np.array([cx, cy, cz, 1])
        rx, ry, rz, _ = np.dot(calibmat, camera_pos)  # unit: m
        robot_locs.append([rx, ry, rz])
    return np.asarray(robot_locs)

def transform_image_to_camera(image_locs, image_width, pc, margins=None):
    """
    1. Replace bad point and adjust the feasible height
    2. Transform image loc to camera loc
    Parameters:
        image_locs {list} -- N * (ix,iy), pixel location
        pc {array} -- (point num x 3) point cloud 
        margins {tupple} -- cropped roi
    Returns: 
        camera_locs {list} -- N * (cx,cy,cz)
    """
    camera_locs = []
    for loc in image_locs:

        (ix, iy) = loc
        if margins == None:
            top_margin, left_margin = 0, 0
        else: 
            top_margin, left_margin = margins["top"], margins["left"]

        full_ix = ix + left_margin
        full_iy = iy + top_margin
        #(new_ix, new_iy) = replace_bad_point(img, (full_ix, full_iy))
        offset = int(full_iy * image_width + full_ix)
        camera_locs.append(pc[offset]) # unit: m
    return camera_locs

def transform_image_to_robot(image_locs, point_cloud, cfg, hand="left", margin=None, tilt=None):
    """
    Transform image locs to robot locs (5-th joint pose in robot coordinate)
    1. p_r_ft -> p_r_j: position of 5-th joint of the arm in robot coordinate
    2. (u,v) -> p_c
    3. p_c -> p_r_f: position of finger tip in robot coordinate
    4. degree -> rpy_r_j: euler angles of 5-th joint in robot coordinate
    Parameters:
        image_locs {array} -- N * (ix,iy,ia) at camera coordinate, angle in degree
        calib_path {str} -- calibration matrix file path
    Returns: 
        robot_locs {array} -- N * (rx,ry,rz,ra) at robot coordinate, angle in degree
    """
    _obj_h = 0.005 
    g_rc = np.loadtxt(cfg["calibmat_path"]) # 4x4, unit: m

    if len(image_locs) == 3: 
        # including calculate euler angle 
        (u, v, theta) = image_locs # theta: degree
        
        # [1]
        if tilt == None: tilt = 90
        if hand == "left":
            rpy_r_j = [(theta - 90), -tilt, 90]
            g_jf = np.array([[1,0,0,-(cfg["hand"]["schunk_length"])],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
        elif hand == "right":
            rpy_r_j = [(theta + 90) % 180, -tilt, -90]
            g_jf = np.array([[1,0,0,-(cfg["hand"]["smc_length"])],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
    elif len(image_locs) == 2: 
        (u, v) = image_locs 
    
    if margin is not None: 
        u += cfg[margin]["margin"]["left"]
        v += cfg[margin]["margin"]["top"]
    
    # [2]
    #(new_ix, new_iy) = replace_bad_point(img, (full_ix, full_iy))
    p_c = point_cloud[int(v * cfg["width"] + u)] # unit: m

    # [3] 
    p_r_f = np.dot(g_rc, [*p_c, 1])  # unit: m
    p_r_f = p_r_f[:3]

    if len(image_locs) == 2: return p_r_f

    # [4]
    p_r_f[2] -= _obj_h
    g_rf = np.r_[np.c_[rpy2mat(rpy_r_j, 'xyz'), p_r_f], [[0,0,0,1]]]
    g_rj = np.dot(g_rf, np.linalg.inv(g_jf))
    p_r_j = np.dot(g_rj, [0,0,0,1])
    
    return p_r_f, [*p_r_j[:3], *rpy_r_j]

def rpy_image_to_robot(degree, hand="left", lr_tilt_degree=(90,60)):
    if hand == "left":
        return [(degree - 90), -lr_tilt_degree[0], 90]
    elif hand == "right":
        return [(degree + 90) % 180, -lr_tilt_degree[1], -90]

def orientation_image_to_robot(degree, hand="left"):
    if hand == "left": 
        return degree - 180 if degree > 90 else degree
    elif hand == "right":
        # return degree - 180 if degree >= 90 else degree
        return (degree + 90) % 180

def check_reachability(robot_loc, min_z, max_z=0.13, min_x=0.30, max_x=0.67, min_y=-0.25, max_y=0.25):
    max_z = 0.2
    (x,y,z) = robot_loc
    if z < min_z or z > max_z:
        warning_print("Out of z-axis reachability! ") 
        return False 
    # elif x < min_x or x > max_x: 
    #     warning_print("Out of x-axis reachability! ")
    #     return False
    # elif y < min_y or y > max_y:
    #     warning_print("Out of y-axis reachability! ")
    #     return False
    else:
        return True

def generate_motion(filepath, ee_pose, action):
    x,y,z,a = ee_pose
    ###### Object: screw ######
    # h_offset = 0.018
    # z -= h_offset 
    ###### Object: harness ######
    # h_offset = 0.013
    # z -= h_offset
    if z < 0.020: 
        warning_print("Touch the plane! ")
        z = 0.020
    z += 0.013
    exec_flag = False
    # for both accuracy and safety, be careful with the robot coordinate
    # if z is too small or too large, DO NOT generate motion file, just empty it
    generator = Motion(filepath)

#     if z < 0.030 or z >=0.130 or x <= 0.30 or x >= 0.644 or y >=0.252 or y <= -0.243: #table
    if not check_reachability((x,y,z), min_z=0.030, max_z=0.130):
        warning_print("Out of robot workspace!! ")
        generator.empty_motion_generator()
        warning_print("Fail! Please try again ... ")

    else:
        exec_flag = True
        if action == 0:
            notice_print("Action scheme No.{}".format(action))
            # generator.motion_generator_dl(x,y,z,a)
            # generator.generate_a_h(x,y,z,a)
            # generator.generate_cone_helix(x,y,z,a)
            # generator.generate_cone_helix_spin(x,y,z,a)
            # generator.generate_diagnal(x,y,z,a)
            # generator.generate_a_tf(x,y,z,a)
        elif action == 1:
            notice_print("Action scheme No.{}".format(action))
            generator.generate_a_h(x,y,z,a)
        elif action == 2:
            notice_print("Action scheme No.{}".format(action))
            generator.generate_a_hs(x,y,z,a)
        elif action == 3:
            notice_print("Action scheme No.{}".format(action))
            generator.generate_a_f(x,y,z,a)
        elif action == 4:
            notice_print("Action scheme No.{}".format(action))
            generator.generate_a_fs(x,y,z,a)
        elif action == 5:
            notice_print("Action scheme No.{}".format(action))
            generator.generate_a_tf(x,y,z,a)
        elif action == 6:
            notice_print("Action scheme No.{}".format(action))
            generator.generate_a_tfs(x,y,z,a)
        else:
            warning_print("No grasp available!! ")
            generator.empty_motion_generator()
            main_proc_print("Fail! Please try again ... ")

    # else:
    #     exec_flag = True
    #     if action == 0:
    #         notice_print("Action scheme No.{}".format(action))
    #         generator.motion_generator_dl(x,y,z,a)
    #         generator.motion_generator_half(x,y,z,a)
    #     elif action == 1:
    #         notice_print("Action scheme No.{}".format(action))
    #         generator.motion_generator_half(x,y,z,a)
    #     elif action == 2:
    #         notice_print("Action scheme No.{}".format(action))
    #         generator.motion_generator_half_spin(x,y,z,a)
    #     elif action == 3:
    #         notice_print("Action scheme No.{}".format(action))
    #         generator.motion_generator_full(x,y,z,a)
    #     elif action == 4:
    #         notice_print("Action scheme No.{}".format(action))
    #         generator.motion_generator_full_spin(x,y,z,a)
    #     elif action == 5:
    #         notice_print("Action scheme No.{}".format(action))
    #         generator.motion_generator_two_full(x,y,z,a)
    #     elif action == 6:
    #         notice_print("Action scheme No.{}".format(action))
    #         generator.motion_generator_two_full_spin(x,y,z,a)
    #     else:
    #         warning_print("No grasp available!! ")
    #         generator.empty_motion_generator()
    #         main_proc_print("Fail! Please try again ... ")
    return exec_flag

