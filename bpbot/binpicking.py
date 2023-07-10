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
from bpbot.motion import *
from bpbot.utils import *

def capture_pc():
    """Capture point cloud

    Returns:
        (array): N x 3, unit: m
    """ 
    import bpbot.device.phoxi.phoxi_client as pclt
    pxc = pclt.PhxClient(host="127.0.0.1:18300")
    pxc.triggerframe()
    pc = pxc.getpcd()
    # gs = pxc.getgrayscaleimg()
    return (pc.copy()/1000) if pc is not None else None

def transform_depth(pcd, mat, max_h, min_h, h, w):
    pc_ = np.c_[pcd, np.ones(pcd.shape[0])]
    pr = np.dot(mat, pc_.T).T

    gray = pr[:,2]
    gray[gray>max_h] = min_h
    gray[gray<min_h] = min_h

    gray = np.reshape(gray, (h, w))
    img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return img

def pc2depth(pcd, cfgdata, container="main"):

    G = np.loadtxt(cfgdata["calibmat_path"])
    pc_ = np.c_[pcd, np.ones(pcd.shape[0])]
    pr = np.dot(G, pc_.T).T

    gray = pr[:,2]
    max_h = cfgdata[container]['height']['max']
    min_h = cfgdata[container]['height']['min']

    gray[gray>max_h] = min_h
    gray[gray<min_h] = min_h
    
    gray = np.reshape(gray, (cfgdata["height"], cfgdata["width"]))
    img = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    img_blur = cv2.medianBlur(img, 5)
    return img, img_blur

def crop_roi(img, cfgdata, container='main', bounding_size=25):

    margins = cfgdata[container]["area"]
    top_margin = margins["top"]
    left_margin = margins["left"]
    bottom_margin = margins["bottom"]
    right_margin = margins["right"]
    # print("top-lft point: ", pr[top_margin*cfgdata["width"]+left_margin])
    # print("btm-rgt point: ", pr[bottom_margin*cfgdata["width"]+right_margin])
    
    # cropped the necessary region (inside the bin)
    img_crop = img[top_margin:bottom_margin, left_margin:right_margin]
    h_, w_= img_crop.shape
    if bounding_size > 0:
        # _s = 25
        cv2.rectangle(img_crop,(0,0),(w_, h_),(0,0,0),bounding_size*2)
    return img_crop

# def crop_roi(img_path, margins=None, bounding=False):
#     """
#     Read image crop roi
#     """

#     img = cv2.imread(img_path)
#     h, w, _ = img.shape
#     if margins is None:
#         (top_margin,left_margin,bottom_margin,right_margin) = (0,0,h,w)
#     else:
#         top_margin = margins["top"]
#         left_margin = margins["left"]
#         bottom_margin = margins["bottom"]
#         right_margin = margins["right"]
    
#     # cropped the necessary region (inside the bin)
#     img_crop = img[top_margin:bottom_margin, left_margin:right_margin]
#     h_, w_, _ = img_crop.shape
#     if bounding == True:
#         _s = 25
#         cv2.rectangle(img_crop,(0,0),(w_, h_),(0,0,0),_s*2)
#     return img_crop

def draw_grasp(grasps, img, h_params=None, top_idx=0, top_only=False, color=(255,255,0), top_color=(0,255,0)):
    """Draw grasps for parallel jaw gripper

    Args:
        grasps (array): N * [x,y,r(degree)]
        img (array): source image
        h_params (dictionary, optional): {"finger_width": _, "finger_length":_,"open_width":_}. Defaults to None.
        top_idx (int, optional): top grasp index among grasps. Defaults to 0.
        top_only (bool, optional): only draw top grasp or not. Defaults to False.
        color (tuple, optional): red. Defaults to (255,0,0).
        top_color (tuple, optional): green. Defaults to (0,255,0).

    Returns:
        array: drawn image
    """
    if isinstance(img, str) and os.path.exists(img):
        img = cv2.imread(img)
    if h_params is not None: 
        finger_w = h_params["finger_width"]
        finger_h = h_params["finger_length"]
        open_w = h_params["open_width"]
    else: 
        # default
        finger_w, finger_h, open_w = 5, 15, 27
    if grasps is None: 
        return img 
    gripper = Gripper(finger_w, finger_h, open_w)
    draw_img = gripper.draw_grasp(grasps, img, color=color, top_color=top_color, top_idx=top_idx, top_only=top_only)
    return draw_img


def draw_pull_grasps(img, g_pull, v_pull, g_hold=None, h_params=None):
    
    if isinstance(img, str) and os.path.exists(img):
        img = cv2.imread(img)
    
    if h_params is not None: 
        finger_w = h_params["finger_width"]
        finger_h = h_params["finger_length"]
        open_w = h_params["open_width"]
    else: 
        # default
        finger_w, finger_h, open_w = 5, 15, 27
    gripper = Gripper(finger_w, finger_h, open_w)
    draw_img = gripper.draw_grasp(g_pull, img, top_color=(0,255,0))
    if g_hold is not None:
        draw_img = gripper.draw_grasp(g_hold, draw_img, top_color=(0,255,255))

    arrow_len=50 
    p = g_pull[:2]
    stop_p = [int(p[0]+v_pull[0]*arrow_len), int(p[1]+v_pull[1]*arrow_len)]
    draw_img = cv2.arrowedLine(draw_img, p, stop_p, (0,255,0), 2)
    # draw_img = draw_vector(img, g_pull[:2], v_pull)
    return draw_img 

def detect_edge(img, t_params, margins=None):
    """
    Read image and detect edge
    """
    from bpbot.tangle_solution import LineDetection

    len_thld = int(t_params["len_thld"])
    dist_thld = int(t_params["dist_thld"])
    sliding_size = t_params["sliding_size"]
    sliding_stride = t_params["sliding_stride"]
    c_size = int(t_params["compressed_size"])
    # (len_thld, dist_thld, sliding_size, sliding_stride, c_size) = t_params
    
    norm_img = cv2.resize(adjust_grayscale(img), (c_size,c_size))

    ld = LineDetection(len_thld=len_thld,dist_thld=dist_thld)
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)
    return drawn, lines_num

def get_entanglement_map(img, t_params):
    """
    Read image and generate entanglement map
    """
    from bpbot.tangle_solution import LineDetection, EntanglementMap
    
    len_thld = int(t_params["len_thld"])
    dist_thld = int(t_params["dist_thld"])
    sliding_size = t_params["sliding_size"]
    sliding_stride = t_params["sliding_stride"]
    c_size = int(t_params["compressed_size"]) 
    norm_img = cv2.resize(adjust_grayscale(img), (c_size,c_size))

    ld = LineDetection(len_thld=len_thld,dist_thld=dist_thld)
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(norm_img, vis=True)
    
    em = EntanglementMap(len_thld, dist_thld, sliding_size, sliding_stride)
    emap, wmat_vis,w,d = em.entanglement_map(norm_img)
    # print("w: {:.2}, d: {:.2}".format(w,d))
    lmap = em.line_map(norm_img)
    bmap = em.brightness_map(norm_img)
    return emap

def picknet(img_path, hand_config):
    img = cv2.imread(img_path)
    hand_attr = [hand_config.get(k) for k in ["finger_width", "finger_length", "open_width"]]
    gripper = Gripper(*hand_attr)
    
    from bpbot.module_picksep import PickSepClient
    psc = PickSepClient()
    ret = psc.infer_picknet(imgpath=img_path)
    if not ret:
        return
    print("Successfully infer grasps by PickNet")
    # ----- temporal add 
    g_pick = gripper.calc_grasp_orientation(img, ret[0][0]) # degree
    print(g_pick)
    return np.array([g_pick])
    # ----- end
    
def pick_or_sep(img_path, hand_config, bin='main'):
    img = cv2.imread(img_path)
    # img = adjust_grayscale(img)
    crop_h, crop_w, _ = img.shape

    left_attr = [hand_config['left'].get(k) for k in ['finger_width', 'finger_length', 'open_width']]
    right_attr = [hand_config['right'].get(k) for k in ['finger_width', 'finger_length', 'open_width']]

    gripper_left = Gripper(*left_attr)
    gripper_right = Gripper(*right_attr)

    from bpbot.module_picksep import PickSepClient
    psc = PickSepClient()

    if bin == 'main':
        if is_bin_empty(img_path): 
            print("Main bin is empty! ")
            return
        ret = psc.infer_picknet(imgpath=img_path)
        if not ret:
            return
        scores_pn = ret[1]
        print(f"[!] Infer main bin: {scores_pn[0]:.3f}, {scores_pn[1]:.3f}")

        # p_pick = ret[0][scores_pn.argmax()]
        if scores_pn[0] < scores_pn[1]:
            p_pick = ret[0][1]
            idx = 1
        elif (scores_pn[0] < 0.3 and scores_pn[1] < 0.3 and scores_pn[0] > scores_pn[1]):
            print("adjust! ")
            p_pick = ret[0][1]
            idx = 1
        else:
            p_pick = ret[0][0]
            idx = 0

        g_pick = gripper_left.point_oriented_grasp(img, p_pick) # degree
        if g_pick is None:
            print("Grasp detection failed for main bin ...")
            return
        # else: 
        #     if scores_pn[0] >= 0.3 and scores_pn[1] > scores_pn[0]: return 0, g_pick
            # else: return 1, g_pick
        # return scores_pn.argmax(), g_pick


        return idx, g_pick

    elif bin == 'buffer': 
        if is_bin_empty(img_path): 
            print("[!] Buffer bin is empty! ")
            return
        
        ret = psc.infer_picknet(img_path)
        if not ret: 
            return
        scores_pn = ret[1]
        print(f"[!] Infer buffer bin: {scores_pn[0]:.3f}, {scores_pn[1]:.3f}")
        # pick or sep condition
        if scores_pn[0] > scores_pn[1]:
        # if scores_pn[0] > scores_pn[1] or (scores_pn[0] < scores_pn[1] and scores_pn[0]>0.55):
            p_pick = ret[0][0]
            g_pick = gripper_left.point_oriented_grasp(img, p_pick) # degree
            if g_pick is None:
                print("Grasp detection failed for buffer bin")
            else: 
                return 0, g_pick
        else:
            ret = psc.infer_pullnet(img_path)
            if not ret:
                return
            p_pull = ret[0]
            v_pull = ret[1]
            g_pull = gripper_left.point_oriented_grasp(img, p_pull)
            if g_pull is None:
                print("Grasp detection failed for buffer bin")
                return
            else:
                return 1, g_pull, v_pull

def gen_motionfile_picksep(mf_path, pose_lft, dest, pulling=None, pose_rgt=None):
    """ Generate motions in motion file format

    Args:
        mf_path (str): path to motion file
        dest (str): destination for only picking, drop/goal
        pose_left (array): [x,y,z,roll,pitch,yaw] left palm pose
        pose_right (array, optional): [x,y,z,roll,pitch,yaw] right palm pose. Defaults to None.
        pulling (array, optional): [pull_x, pull_y, pull_len]. Defaults to None.
    """
    arm = "L"
    actor_pick = PickAndPlaceActor(mf_path, arm)
    actor_pull = PullActor(mf_path, arm)

    if pulling is None:
        actor_pick.get_action(pose_lft, dest=dest)

    elif pose_rgt is None:
        actor_pull.get_action(pose_lft, pulling, wiggle=True)
    
    else:
        print("[!] Wrong type for motion generator ...")    

def gen_motionfile_pick(mf_path, eef_pose, arm="left", action_idx=0):
    """Generate motion in motion file format for single-arm picking

    Args:
        mf_path (str): motion file path
        pose_left (array): [x,y,z,roll,pitch,yaw]
        sub_action (int): -1(no motion),0(direct lifting),1,2,3,4,5,6
    """
    arm = arm[0].upper()
    action_name = ["a_dl","a_h","a_hs","a_f","a_fs","a_tf","a_tfs"]
    print(f"Generate motion for {action_name[action_idx]} ... ")
    if action_idx == 0:
        actor = PickAndPlaceActor(mf_path, arm)
        actor.get_action(eef_pose)
    else:
        actor = HelixActor(mf_path)
        actor.get_action(eef_pose, action_idx)
        # actor = PullActor(mf_path, arm)
        # actor.get_action(pose_left, v=[1,0,0.01,0.1], wiggle=True)

def detect_grasp_orientation(p, img_path, g_params, h_params):
    img = cv2.imread(img_path)
    
    gripper_attr = [h_params.get(k) for k in ["finger_width", "finger_length", "open_width"]]
    gripper = Gripper(*gripper_attr)
    g = gripper.point_oriented_grasp(img, p)
    return g
    
def detect_grasp(n_grasp, img_path, g_params, h_params):
    """Detect grasp point using fast graspability evaluation

    Args:
        n_grasp (int): number of grasps you want to output
        img_path (str): image path 
        g_params (dict): {"finger_width":0, "finger_length":0, "open_width":0}
        h_params (dict): {"rotation_step":0, "depth_step": 0, "hand_depth":0}

    Returns:
        (array): grasps = n_grasp * [x,y,r(degree)], return None if no grasp detected
    """
    img = cv2.imread(img_path)
    img_adj = img
    # img_adj = adjust_grayscale(img)
    height, width, _ = img.shape
    
    finger_w = h_params["finger_width"]
    finger_h = h_params["finger_length"]
    # open_w = h_params["open_width"] + random.rand'int(0, 5)
    open_w = h_params["open_width"]
    r2p = h_params["real2pixel"]

    gripper = Gripper(finger_w=finger_w, 
                      finger_h=finger_h, 
                      open_w=open_w,
                      real2pixel=r2p)
    # # gripper.tplt_size = max(height, width, (finger_w*2 + open_w))

    hand_open_mask, hand_close_mask = gripper.create_hand_model()

    # hand_open_mask, hand_close_mask = gripper.create_suction_model()
    rstep = g_params["rotation_step"]
    dstep = g_params["depth_step"]
    hand_depth = g_params["hand_depth"]
    method = Graspability(rotation_step=rstep, 
                          depth_step=dstep, 
                          hand_depth=hand_depth)

    # generate graspability map
    candidates = method.graspability_map(img_adj, 
                                         hand_open_mask=hand_open_mask, 
                                         hand_close_mask=hand_close_mask)
    
    if candidates != []:
    # rank grasps
        # grasps = method.grasp_ranking(candidates, n=n_grasp, h=height, w=width, _dismiss=gripper.open_w/2+gripper.finger_w*2+70)
        grasps = method.grasp_ranking(candidates, n=n_grasp, h=height, w=width)
        return grasps

    return

def detect_nontangle_grasp(n_grasp, img_path, g_params, h_params, t_params):
    """Detect grasp point using fast graspability evaluation

    Args:
        n_grasp (int): number of grasps you want to output
        img_path (str): image path 
        g_params (dict): {"finger_width":0, "finger_length":0, "open_width":0}
        h_params (dict): {"rotation_step":0, "depth_step": 0, "hand_depth":0}
        t_params (dict): {"compressed_size":0, "len_thld": 0, "dist_thld":0, "sliding_size":0, "sliding_stride":0}

    Returns:
        (array): grasps = n_grasp * [x,y,r(degree)], return None if no grasp detected
    """
    img = cv2.imread(img_path)
    img = adjust_grayscale(img)
    plt.imshow(img), plt.show()

    emap = get_entanglement_map(img, t_params)

    height, width, _ = img.shape
    # _ssz = int(t_params["sliding_size"] * height / t_params["compressed_size"])
    # _sst = int(t_params["sliding_stride"] * height / t_params["compressed_size"])
    _ssz = 175
    _sst = 50
    
    # emap /= (emap.max() / 255.0) # real writhe value
    emap = cv2.normalize(emap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    emap = cv2.resize(emap, (width, height))
    region_min = 9999
    region_oi = None
    roi_left_top = [0, 0]
    for y in range(0,height-_ssz + 1, _sst): 
        for x in range(0,width-_ssz + 1, _sst): 
            cropped = emap[y:y + _ssz , x:x + _ssz]
            if cropped.mean() <= region_min: 
                region_min = cropped.mean()

                region_oi = img[y:y + _ssz , x:x + _ssz]
                roi_left_top = [x, y]
    finger_w = int(h_params["finger_width"] * _ssz / width)
    finger_h = int(h_params["finger_length"] * _ssz / width)
    open_w = int(h_params["open_width"] * _ssz / width)
    
    gripper = Gripper(finger_w=finger_w, 
                      finger_h=finger_h, 
                      open_w=open_w)
    
    rstep = g_params["rotation_step"]
    dstep = g_params["depth_step"] + random.randint(0, 5)
    hand_depth = g_params["hand_depth"]
    
    method = Graspability(rotation_step=rstep, 
                          depth_step=dstep, 
                          hand_depth=hand_depth)
    
    hand_open_mask, hand_close_mask = gripper.create_hand_model()

    # generate graspability map
    
    candidates = method.graspability_map(region_oi, 
                                         hand_open_mask=hand_open_mask, 
                                         hand_close_mask=hand_close_mask)
    
    loc = (int(candidates[0][1]), int(candidates[0][2]))
    cv2.circle(region_oi, loc, 5, (0,255,0), -1)
    plt.imshow(region_oi), plt.show()
    # candidates = method.combined_graspability_map(img, hand_open_mask, hand_close_mask, emap)
    # roi -> complete depth map
    if candidates != []:
        candidates[:,1:3] += roi_left_top
        # ranking grasps
        grasps = method.grasp_ranking(candidates, n=n_grasp, h=height, w=width)
        return grasps, emap
    return

def predict_action_grasp(grasps, imgpath):
    """Predict the best action grasp from candidates pairs

    Args:
        grasps (array): [[x1,y1],[x2,y2]]
        crop_path (str): path of cropped image

    Returns:
        (int, int): index of actions / grasps
    """
    from bpbot.module_asp import asp_client as aspclt
    aspc = aspclt.ASPClient()
    
    # res_p = aspc.predict(imgpath=imgpath, grasps=grasps)

    best_action,best_graspno = aspc.infer(imgpath=imgpath, grasps=grasps)
    #grasps2bytes=np.ndarray.tobytes(np.array(grasps))
    #predict_result= aspc.predict(imgpath=crop_path, grasps=grasps2bytes)
    #best_action = predict_result.action
    #best_graspno = predict_result.graspno
    return best_action, best_graspno

##################### TODO: test following code ##################
def detect_grasp_width_adjusted(n_grasp, img_path, margins, g_params, h_params):
    """detect grasp points with adjusting width"""
    (top_margin,left_margin,bottom_margin,right_margin) = margins
    img = cv2.imread(img_path)

    # cropped the necessary region (inside the bin)
    height, width, _ = img.shape
    img_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = img_cut.shape
    print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    
    im_adj = adjust_grayscale(img_cut)
    im_adj = img_cut
    (finger_w, finger_h, open_w, tplt_size) = h_params

    min_open_w = 25
    open_step = 20
    
    all_candidates = []

    while open_w >= min_open_w:
        # ------------------
        gripper = Gripper(finger_w=finger_w, finger_h=finger_h, open_w=open_w, tplt_size=tplt_size)
        hand_open_mask, hand_close_mask = gripper.create_hand_model()

        (rstep, dstep, hand_depth) = g_params
        method = Graspability(rotation_step=rstep, depth_step=dstep, hand_depth=hand_depth)

        # generate graspability map
        print("Generate graspability map  ... ")
        candidates = method.width_adjusted_graspability_map(
            im_adj, hand_open_mask=hand_open_mask, hand_close_mask=hand_close_mask,width_count=open_w)
        all_candidates += candidates
        # ------------------
        open_w -= open_step

    if all_candidates != []:
    # detect grasps
        print("Detect grasp poses ... ")
        grasps = method.grasp_ranking(
            all_candidates, n=n_grasp, h=cropped_height, w=cropped_width)
        # print(grasps)
        if grasps != [] :
            print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            drawn_input_img = gripper.draw_grasp(grasps, im_adj.copy(), (73,192,236))
            # cv2.imwrite("/home/xinyi/Pictures/g_max_pixel_area.png", drawn_input_img)
            cv2.imshow("grasps", drawn_input_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            return grasps, im_adj, img
    else:
        print("[!] Grasp detection failed! No grasps!")
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
    img_cut = img[top_margin:bottom_margin, left_margin:right_margin]
    cropped_height, cropped_width, _ = img_cut.shape
    print("Crop depth map to shape=({}, {})".format(cropped_width, cropped_height))
    # im_adj = adjust_grayscale(img_cut)
    im_adj = img_cut

    # create gripper
    (finger_w, finger_h, open_w, tplt_size) = h_params
    gripper = Gripper(finger_w=finger_w, finger_h=finger_h, open_w=open_w, tplt_size=tplt_size)
    hand_open_mask, hand_close_mask = gripper.create_hand_model()

    (rstep, dstep, hand_depth) = g_params
    method = Graspability(rotation_step=rstep, depth_step=dstep, hand_depth=hand_depth)

    # generate graspability map
    all_candidates = []
    for d in np.arange(0, 201, 50):
        _, Wt = cv2.threshold(touch_mask, d + GripperD, 255, cv2.THRESH_BINARY)
        _, Wc = cv2.threshold(depth, d, 255, cv2.THRESH_BINARY)

        # Wc = cv2.bitwise_or(Wc, cv2.subtract(touch_mask, Wt))
        print("Generate graspability map  ... ")
        candidates = method.target_oriented_graspability_map(
            im_adj, hand_open_mask=hand_open_mask, hand_close_mask=hand_close_mask,
            Wc=Wc, Wt=Wt)

        all_candidates += candidates
    
    # detect grasps

    if all_candidates != []:
    # detect grasps
        print("Detect grasp poses ... ")
        grasps = method.grasp_ranking(
            all_candidates, n=n_grasp, h=cropped_height, w=cropped_width, _distance=50)
        # print(grasps)
        if grasps != [] :
            print(f"Success! Detect {len(grasps)} grasps from {len(candidates)} candidates! ")
            # draw grasps
            drawn_input_img = gripper.draw_grasp(grasps, im_adj.copy(), (73,192,236))
            # cv2.imwrite("/home/xinyi/Pictures/g_max_pixel_area.png", drawn_input_img)
            cv2.imshow("grasps", drawn_input_img)
            cv2.waitKey()
            cv2.destroyAllWindows()
            return grasps, im_adj, img
    else:
        print("[!] Grasp detection failed! No grasps!")
        return None, im_adj,img

def is_bin_empty(img_path):
    low_thld = 20
    high_thld = 90
    pixel_thld = 300
    
    img = cv2.imread(img_path)
    edge = cv2.Canny(img, low_thld, high_thld)
    # print("edge", np.count_nonzero(edge), "threshold: ", pixel_thld) 
    # plt.imshow(edge), plt.show()
    if np.count_nonzero(edge) < pixel_thld: 
        return True
    else: 
        return False

def is_colliding(p, v, cfgdata, point_cloud, margin="drop"):
    m = cfgdata[margin]["area"]
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

def transform_force_sensor_to_robot(p_fs):
    p_r = np.array(p_fs)
    p_r[:,1] *= -1
    return p_r

def transform_image_to_robot(image_locs, point_array, cfgdata, arm='left', container='main', tilt=None):
    """
    Transform image locs to robot locs (5-th joint pose in robot coordinate)
    1. p_tcpt -> p_wrist: position of 5-th joint of the arm in robot coordinate
    2. (u,v) -> p_c
    3. p_c -> p_tcp: position of finger tip in robot coordinate
    4. degree -> rpy_wrist: euler angles of 5-th joint in robot coordinate
    Parameters:
        image_locs {array} -- N * (ix,iy,ia) at camera coordinate, angle in degree
        calib_path {str} -- calibration matrix file path
    Returns: 
        robot_locs {array} -- N * (rx,ry,rz,ra) at robot coordinate, angle in degree
    """
    _obj_h = cfgdata["obj_height"] / 1000
    # _obj_h += 0.005 
    # _obj_h += 0.01 
    g_rc = np.loadtxt(cfgdata["calibmat_path"]) # 4x4, unit: m
    # if container == "pick":
    #     g_rc = np.loadtxt(cfgdata["calibmat_path"]) # 4x4, unit: m
    # else:
    #     # print("load new matrix")
    #     g_rc = np.loadtxt("/home/hlab/bpbot/data/calibration/calibmat_d.txt") # 4x4, unit: m
    if len(image_locs) == 3: 
        # including calculate euler angle 
        (u, v, theta) = image_locs # theta: degree
        
        # [1]
        if tilt == None: tilt = 90
        if arm == "left":
            g_wt = np.array([[1,0,0,-(cfgdata["hand"]["left"]["height"])],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
            # rpy_wrist = [(theta - 90), -tilt, 90]
            rpy_wrist = [-((90-theta)%180), -tilt, 90]
            # if dualarm is False: 
            #     rpy_wrist = [90), -tilt, 90]
            # else: 
            #     rpy_wrist = [-(theta - 90) % -180, -tilt, 90]

        elif arm == "right":
            rpy_wrist = [(theta + 90) % 180, -tilt, -90]
            g_wt = np.array([[1,0,0,-(cfgdata["hand"]["right"]["height"])],
                            [0,1,0,0],
                            [0,0,1,0],
                            [0,0,0,1]])
    elif len(image_locs) == 2: 
        (u, v) = image_locs 
    
    if container is not None: 
        u += cfgdata[container]["area"]["left"]
        v += cfgdata[container]["area"]["top"]
    
    u = int(u)
    v = int(v)
    # [2]
    point_mat = np.reshape(point_array, (cfgdata["height"],cfgdata["width"],3))
    p_c = point_mat[v,u]
    # print("camera_pose", p_c)
    #p_c = point_array[int(v * cfgdata["width"] + u)] # unit: m
    # if (p_c == 0).all(): :w

    #     u,v = replace_bad_point(point_mat[:,:,2], [u,v], "min", 25)
    u,v = replace_bad_point(point_mat[:,:,2], [u,v], "min", 25)
    p_c = point_mat[v,u]
    # print("==> new camera_pose", p_c)
    #if np.count_nonzero(p_c) == 0:
    #    print("replace bad point! ")
    #    u,v = replace_bad_point(point_mat[:,:,2], [u,v])
    #    p_c = point_mat[v,u]
    # print("new camera_pose", p_c)
    # [3] 
    p_tcp = np.dot(g_rc, [*p_c, 1])  # unit: m
    p_tcp = p_tcp[:3]
    
    # p_tcp_table = np.dot(g_rc, [*p_c[0:2], cfgdata["table_distance"]/1000, 1])
    p_tcp[2] -= _obj_h

    if len(image_locs) == 2: return p_tcp

    # [4]
    g_rt = np.r_[np.c_[rpy2mat(rpy_wrist, 'xyz'), p_tcp], [[0,0,0,1]]]
    g_rw = np.dot(g_rt, np.linalg.inv(g_wt))
    p_wrist = np.dot(g_rw, [0,0,0,1])

    return p_tcp, [*p_wrist[:3], *rpy_wrist]

def check_reachability(robot_loc, min_z, max_z=0.13, min_x=0.30, max_x=0.67, min_y=-0.25, max_y=0.25):
    max_z = 0.2
    (x,y,z) = robot_loc
    if z < min_z or z > max_z:
        print("[!] Out of z-axis reachability! ") 
        return False 
    # elif x < min_x or x > max_x: 
    #     print("[!] Out of x-axis reachability! ")
    #     return False
    # elif y < min_y or y > max_y:
    #     print("[!] Out of y-axis reachability! ")
    #     return False
    else:
        return True

def check_force_file():
    threshold = 0.07
    init = [7946,8256,8563]
    N = 20
    force = np.loadtxt(open('/home/hlab/bpbot/bpbot/driver/dynpick/out.txt','rt').readlines()[:-1])
    #force = np.loadtxt("/home/hlab/bpbot/bpbot/driver/dynpick/out.txt")
    f = (force[-N:][:,1:4]-init) / 1000
    print(f)
    for i in range(N-1):
        if (f[i+1][2] - f[i][2]) > 0.01 and f[i+1][2] > threshold: 
            print("tangle!!!")
            return True
    return False
    
def check_force(duration=30000, threshold=0.1):
    # find port of sensor and open
    import glob
    import termios
    import time
    try:
        port = (glob.glob(r'/dev/ttyUSB*'))[0]
        #os.system("sudo chmod a+rw " + port)
        fdc = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        print("Open port "+port)
    except BlockingIOError as e:
        print("Can't open port! ")
        fdc = -1

    if (fdc < 0):
        os.close(fdc)
    ############# tty control ################
    term_ = termios.tcgetattr(fdc)
    term_[0] = termios.IGNPAR #iflag
    term_[1] = 0 # oflag
    term_[2] = termios.B921600 | termios.CS8 | termios.CLOCAL | termios.CREAD # cflag
    term_[3] = 0 # lflag -> ICANON
    term_[4] = 4103 # ispeed
    term_[5] = 4103 # ospeed
    # # cc
    o_ = bytes([0])
    term_[6][termios.VINTR] = o_ # Ctrl-c
    term_[6][termios.VQUIT] = o_ # Ctrl-?
    term_[6][termios.VERASE] = o_ # del
    term_[6][termios.VKILL] = o_ # @
    term_[6][termios.VEOF] =  bytes([4])# Ctrl-d
    term_[6][termios.VTIME] = 0
    term_[6][termios.VMIN] = 0
    term_[6][termios.VSWTC] = o_ # ?0
    term_[6][termios.VSTART] = o_ # Ctrl-q
    term_[6][termios.VSTOP] = o_ # Ctrl-s
    term_[6][termios.VSUSP] = o_ # Ctrl-z
    term_[6][termios.VEOF] = o_ # ?0
    term_[6][termios.VREPRINT] = o_ # Ctrl-r
    term_[6][termios.VDISCARD] = o_ # Ctrl-u
    term_[6][termios.VWERASE] = o_ # Ctrl-w
    term_[6][termios.VLNEXT] = o_ # Ctrl-v
    term_[6][termios.VEOL2] = o_ # ?0

    termios.tcsetattr(fdc, termios.TCSANOW, term_)
    ################## over ##################

    #tw = 50
    tw = 100
    clkb = 0
    clkb2 = 0
    num = 0
    clk0 = (time.process_time())*1000 # ms

    r_ = str.encode("R")
    os.write(fdc, r_)

    # initialization
    #init = [7984,8292,8572]
    init = [7946,8256,8563]
    #fp = open('./out.txt','wt')
    plot_data = []
    j = 0
    time_stamp = 0
    
    while time_stamp < duration:
        # half second
        try:
            data = []
            while True:
                clk = (time.process_time()) * 1000 - clk0
                if clk >= (clkb + tw):
                    clkb = clk / tw * tw
                    break
            os.write(fdc, r_)
            l = os.read(fdc, 27)
            time_stamp = int(clk / tw * tw)
            if l == bytes():
                continue

            data.append(time_stamp)
            for i in range(1,22,4):
                data.append(int((l[i:i+4]).decode(),16))
            #fp.write(",".join(map(str,data)))
            #fp.write("\n")

            force = [(data[1]-init[0])/1000, (data[2]-init[1])/1000, (data[3]-init[2])/1000]
            print(force)
            if len(plot_data) > 1 and (force[2]>0 and plot_data[-1][2] >0)and (force[2] -  plot_data[-1][2]) > 0.03 and force[2] > threshold: 
                print("tangle!!!")
                
                return True
            plot_data.append([(data[1]-init[0])/1000, (data[2]-init[1])/1000, (data[3]-init[2])/1000])
            
        except KeyboardInterrupt:
            break
    print("No tangle! ")
    return False

def plot_f(force, x):
    colors = [[ 78.0/255.0,121.0/255.0,167.0/255.0], # 0_blue
              [255.0/255.0, 87.0/255.0, 89.0/255.0], # 1_red
              [ 89.0/255.0,169.0/255.0, 79.0/255.0], # 2_green
              [237.0/255.0,201.0/255.0, 72.0/255.0], # 3_yellow
              [242.0/255.0,142.0/255.0, 43.0/255.0], # 4_orange
              [176.0/255.0,122.0/255.0,161.0/255.0], # 5_purple
              [255.0/255.0,157.0/255.0,167.0/255.0], # 6_pink
              [118.0/255.0,183.0/255.0,178.0/255.0], # 7_cyan
              [156.0/255.0,117.0/255.0, 95.0/255.0], # 8_brown
              [186.0/255.0,176.0/255.0,172.0/255.0]] # 9_gray

    f_new = np.asarray(force)
    fig = plt.figure(1, figsize=(16, 6))
    ax1 = fig.add_subplot(111)
    # ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_prop_cycle(color=colors)
    #major_ticks = np.arange(x[0], x[-1], 10 if len(x) < 160 else 20)
    #minor_ticks = np.arange(x[0], x[-1], 1 if len(x) < 160 else 2)
    major_ticks = np.arange(x[0], x[-1], 10)
    
    #minor_ticks = np.arange(x[0], x[-1])
    hline_c = 'gold'

    for i, f in enumerate(f_new):
        if i==0: continue
        if f[2] > f_new[i-1][2] and f[2] > 0.1:
            ax1.axvline(x=x[i], color=hline_c, alpha=1)
            ax1.axhline(y=0, color=colors[0], alpha=.5, linestyle='dashed')
            print("Oops! ")
        #if abs(f[1]) > 4.8:
        #    ax1.axvline(x=x[i], color=hline_c, alpha=1)
        #    ax1.axhline(y=4.8 if f[1] > 0 else -4.8, color=colors[1], alpha=.5, linestyle='dashed')
        #    print(i)
        #if abs(f[2]) > 6:
        #    ax1.axvline(x=x[i], color=hline_c, alpha=1)
        #    ax1.axhline(y=6 if f[2] > 0 else -6, color=colors[2], alpha=.5, linestyle='dashed')
        #    print(i)
    #ax1.axhline(y=4, color=colors[0], alpha=.5, linestyle='dashed')
    ax1.set_title('Force')
    ax1.set_xticks(major_ticks)

    #ax1.set_xticks(minor_ticks, minor=True)
    # ax1.axhspan(0, 6, facecolor=colors[0], alpha=.1)
    ax1.grid(which='minor', linestyle='dotted', alpha=.5)
    ax1.grid(which='major', linestyle='dotted', alpha=1)
    ax1.plot(x, f_new, label=['Fx', 'Fy', 'Fz'])
    ax1.legend()
    #handles, labels = ax1.get_legend_handles_labels()
    #ax1.legend(handles=handles, labels=eval(labels[0]), loc='upper left')
    plt.ion()
    plt.ylim(-2, 2)
    plt.show()



