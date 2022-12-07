import os
import random
import importlib
spec = importlib.util.find_spec("cnoid")
FOUND_CNOID = spec is not None
if FOUND_CNOID: 
    from cnoid.Util import *
    from cnoid.Base import *
    from cnoid.Body import *
    from cnoid.BodyPlugin import *
    from cnoid.GraspPlugin import *
    from cnoid.BinPicking import *
    topdir = executableTopDirectory
else: 
    topdir = "/home/hlab/choreonoid-1.7.0/"

from bpbot.binpicking import *
from bpbot.config import BinConfig
from bpbot.robotcon.nxt.nxtrobot_client import NxtRobot
from bpbot.utils import * 
import timeit
import numpy as np
start = timeit.default_timer()

# ========================== define path =============================
mode = 1
# 0: pull, 1: pull + direction
# 2: hold and pull

# get root dir
#root_dir = os.path.abspath("./")
root_dir = os.path.join(topdir, "ext/bpbot")
#root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
print(f"[*] Start at {root_dir} ")

depth_dir = os.path.join(root_dir, "data/depth/")

img_path = os.path.join(root_dir, "data/depth/depth.png")
crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")
calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
traj_path = os.path.join(root_dir, "data/motion/motion_ik.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")


# ======================= get config info ============================

cfg = BinConfig(config_path)
cfg = cfg.data

# ======================== get depth img =============================

point_array = capture_pc()
if point_array is None: 
    print("[!] Exit! ")
    sys.exit()

img, img_blur = pc2depth(point_array, cfgdata["drop"]["height"], cfgdata["width"], cfgdata["height"])
point_array /= 1000
cv2.imwrite(img_path, img_blur)
# import open3d as o3d
# pcd = o3d.io.read_point_cloud("/home/hlab/Desktop/test_ply.ply")
# point_array = pcd.points
img_input = crop_roi(img_path, margins=cfgdata["drop"]["area"])
cv2.imwrite(crop_path, img_input)

# =======================  compute grasp =============================

drawn = img_input.copy()
p_clicked = [] # 2x2, [[pull_x, pull_y], [hold-x, hold_y]]
def on_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(drawn,(x,y),5,(0,255,0),-1)
        # print(f"{x},{y}")
        p_clicked.append([x,y])


# p_clicked = [[102,107], [308, 243]]
if mode == 0: 
    cv2.namedWindow('Click to select pull')
    cv2.setMouseCallback('Click to select pull',on_click)
    while(len(p_clicked)<1):
        cv2.imshow('Click to select pull',drawn)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k==ord('q'):
            break
    cv2.destroyAllWindows()
    p_pull = np.array(p_clicked)[0]
    v_pull = np.array([1,0])
    print(f"Mode 0: p_pull={p_pull}, v_pull={v_pull}")

elif mode == 1:
    cv2.namedWindow('Click to select pull position&direction')
    cv2.setMouseCallback('Click to select pull position&direction',on_click)
    while(len(p_clicked)<2):
        cv2.imshow('Click to select pull position&direction',drawn)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k==ord('q'):
            break
    cv2.destroyAllWindows()
    p_pull, p_end = np.array(p_clicked)
    v_pull = p_end - p_pull # image  
    v_pull = v_pull / np.linalg.norm(v_pull)
    print(f"Mode 1: p_pull={p_pull}, v_pull={v_pull}")

elif mode == 2:
    cv2.namedWindow('Click to select hold and pull')
    cv2.setMouseCallback('Click to select hold and pull',on_click)
    while(len(p_clicked)<2):
        cv2.imshow('Click to select hold and pull',drawn)
        k = cv2.waitKey(20) & 0xFF
        if k == 27 or k==ord('q'):
            break
    cv2.destroyAllWindows()
    p_hold, p_pull = np.array(p_clicked)
    v_pull = np.array([1,0])
    print(f"Mode 2: p_hold={p_hold}, p_pull={p_pull}, v_pull={v_pull}")

# =======================  picking policy ===========================
v_len = 0.12
right_attr = [cfgdata["hand"]["right"].get(k) for k in ["finger_width", "finger_length", "open_width"]]
left_attr = [cfgdata["hand"]["left"].get(k) for k in ["finger_width", "finger_length", "open_width"]]

gripper_right = Gripper(*right_attr)
gripper_left = Gripper(*left_attr)
# theta_pull = gripper_left.point_oriented_grasp(img_input, [p_pull[0], p_pull[1]])
g_pull = gripper_left.point_oriented_grasp(img_input, p_pull)
crop_grasp = draw_pull_grasps(img_input.copy(), g_pull, v_pull)
# crop_grasp = gripper_left.draw_grasp(g_pull, img_input.copy())
p_tcp_pull, g_wrist_pull = transform_image_to_robot(g_pull, point_array, cfg, hand="left", margin="drop",dualarm=True)
v_tcp_pull = [v_pull[1],v_pull[0]]
if mode == 2: 
    g_hold = gripper_left.point_oriented_grasp(img_input, [p_hold[0], p_pull[1]])
    # theta_hold = 90
    # crop_grasp = gripper_left.draw_grasp(g_pull, img_input.copy())
    crop_grasp = gripper_right.draw_grasp(g_hold, crop_grasp)
    p_tcp_hold, g_wrist_hold = transform_image_to_robot(g_hold, point_array, cfg, hand="right", margin="drop", tilt=60, dualarm=True)
    notice_print("Grasp (hold): (%d,%d,%.1f) -> TCP (%.3f,%.3f,%.3f)" % (*g_hold, *p_tcp_hold))
    notice_print("Grasp (pull): (%d,%d,%.1f) -> TCP (%.3f,%.3f,%.3f)" % (*g_pull, *p_tcp_pull))
    notice_print("Vector (pull): (%.2f,%.2f), length: %.3f" % (*v_pull, v_len))
    gen_motion_pickorsep(mf_path, g_wrist_pull, pose_right=g_wrist_hold, pulling=[*v_tcp_pull,v_len])
else: 
    notice_print("Grasp (pull): (%d,%d,%.1f) -> TCP (%.3f,%.3f,%.3f)" % (*g_pull, *p_tcp_pull))
    notice_print("Vector (pull): (%.2f,%.2f), length: %.3f" % (*v_pull, v_len))
    gen_motion_pickorsep(mf_path, g_wrist_pull, pulling=[*v_tcp_pull,v_len])

cv2.imshow("", crop_grasp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# v_len = check_collision(p_r_pull[:2], v_pull, cfg, point_array)

print("[*] Grasp (pull): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                % (*g_pull, *g_r_pull))
print("[*] Grasp (hold): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                % (*g_hold, *g_r_hold))
print("[*] Vector (pull): (%.2f,%.2f), length: %.3f" % (*v_pull, v_len))

# # =======================  generate motion ===========================
if FOUND_CNOID: 
    plan_success = load_motionfile(mf_path)
    print("plannning success? ", plan_success)
    nxt = NxtRobot(host='[::]:15005')
    motion_seq = get_motion()
    num_seq = int(len(motion_seq)/20)
    print(f"[*] Total {num_seq} motion sequences! ")
    motion_seq = np.reshape(motion_seq, (num_seq, 20))

    nxt.playMotionSeq(motion_seq)

# if FOUND_CNOID: 
#     # gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra], dest="drop")
#     # gen_success = generate_motion(mf_path, [rx,ry,rz,ra], best_action)
#     plan_success = load_motionfile(mf_path)
#     # if gen_success and plan_success:
#     if plan_success:
#         nxt = NxtRobot(host='[::]:15005')
#         motion_seq = get_motion()
#         num_seq = int(len(motion_seq)/21)
#         print(f"Total {num_seq} motion sequences! ")
#         motion_seq = np.reshape(motion_seq, (num_seq, 21))

#         # nxt.playMotionSeq(motion_seq)    
# # ======================= Record the data ===================s=========

# end = timeit.default_timer()
# print("[*] Time: {:.2f}s".format(end - start))
