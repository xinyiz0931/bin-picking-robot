import os
import random
import importlib
spec = importlib.util.find_spec("cnoid")
found_cnoid = spec is not None
if found_cnoid: 
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

# get root dir
#root_dir = os.path.abspath("./")
root_dir = os.path.join(topdir, "ext/bpbot")
#root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
main_proc_print(f"Start at {root_dir} ")

depth_dir = os.path.join(root_dir, "data/depth/")

img_path = os.path.join(root_dir, "data/depth/depth.png")
crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")
calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
traj_path = os.path.join(root_dir, "data/motion/motion_ik.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")


# ======================= get config info ============================

bincfg = BinConfig(config_path)
cfg = bincfg.data

# ======================== get depth img =============================

point_array = get_point_cloud(depth_dir, cfg['mid']['distance'],
                                 cfg['width'],cfg['height'])
point_array /= 1000
# import open3d as o3d
# pcd = o3d.io.read_point_cloud("/home/hlab/Desktop/test_ply.ply")
# point_array = pcd.points
img_input = crop_roi(img_path, margins=cfg["mid"]["margin"])
cv2.imwrite(crop_path, img_input)

# =======================  compute grasp =============================

drawn = img_input.copy()
p_clicked = [] # 2x2, [[pull_x, pull_y], [hold-x, hold_y]]
def on_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(drawn,(x,y),5,(0,255,0),-1)
        # print(f"{x},{y}")
        p_clicked.append([x,y])

cv2.namedWindow('click twice to select pull and hold')
cv2.setMouseCallback('click twice to select pull and hold',on_click)
while(len(p_clicked)<2):
    cv2.imshow('click twice to select pull and hold',drawn)
    k = cv2.waitKey(20) & 0xFF
    if k == 27 or k==ord('q'):
        break
cv2.destroyAllWindows()

# p_clicked = [[102,107], [308, 243]]
p_hold, p_pull = np.array(p_clicked)

# =======================  picking policy ===========================


smc_attr = [cfg["hand"]["smc"].get(k) for k in ["finger_width", "finger_height", "open_width"]]
schunk_attr = [cfg["hand"]["schunk"].get(k) for k in ["finger_width", "finger_height", "open_width"]]

gripper_smc = Gripper(*smc_attr)
gripper_schunk = Gripper(*schunk_attr)
theta_pull = gripper_schunk.point_oriented_grasp(img_input, [p_pull[0], p_pull[1]])
grasp_pull = [p_pull[0], p_pull[1], theta_pull*math.pi/180]

# theta_hold = gripper_schunk.point_oriented_grasp(img_input, [p_hold[0], p_pull[1]])
theta_hold = 90
grasp_hold = [p_hold[0], p_hold[1], theta_hold*math.pi/180]
crop_grasp = gripper_schunk.draw_grasp(grasp_pull, img_input.copy())
crop_grasp = gripper_smc.draw_grasp(grasp_hold, crop_grasp)

cv2.imshow("", crop_grasp)
cv2.waitKey(0)
cv2.destroyAllWindows()

p_r_pull, g_pull = transform_image_to_robot([*p_pull, theta_pull], point_array, cfg, hand="left", margin="mid")
p_r_hold, g_hold = transform_image_to_robot([*p_hold, theta_hold], point_array, cfg, hand="right", margin="mid", tilt=60)

# obj_h = 0.0
# # using calibmat to transform image location to fingertip location
# [p_r_pull, p_r_hold] = transform_image_to_robot([p_pull, p_hold], cfg["width"], calib_path, point_array, margins=cfg["mid"]["margin"])
# theta_r_pull = orientation_image_to_robot(theta_pull, hand="left")
# theta_r_hold = orientation_image_to_robot(theta_hold, hand="right")
# # rpy in degree
# rpy_r_pull = rpy_image_to_robot(theta_pull, hand="left")
# rpy_r_hold = rpy_image_to_robot(theta_hold, hand="right")
# print(f"Pull: {rpy_r_pull}, Hold: {rpy_r_hold}")

# p_j_pull = transform_tmp(p_r_pull, rpy_r_pull, cfg["hand"]["schunk_length"] - obj_h)
# p_j_hold = transform_tmp(p_r_hold, rpy_r_hold, cfg["hand"]["smc_length"] - obj_h)
# print(f"Pull: {p_j_pull}, Hold: {p_j_hold}")


v_pull = np.array([-1,0]) # robot coordinate
# v_len = check_collision(p_r_pull[:2], v_pull, cfg["mid"]["margin"], cfg["width"], calib_path, point_array)
v_len = check_collision(p_r_pull[:2], v_pull, cfg, point_array)

notice_print("Pull point: (%d,%d) -> joint (%.3f,%.3f,%.3f), degree=%.1f" 
            % (p_pull[0], p_pull[1], g_pull[0], g_pull[1], g_pull[2], theta_pull))
notice_print("Pull vector: (%.2f,%.2f), length: %.3f" % (v_pull[0], v_pull[1], v_len))

notice_print("Hold point: (%d,%d) -> joint (%.3f,%.3f,%.3f), degree=%.1f" 
            % (p_hold[0], p_hold[1], g_hold[0], g_hold[1], g_hold[2], theta_hold))
# r_pull = np.append(p_j_pull, rpy_r_pull)
# r_hold = np.append(p_j_hold, rpy_r_hold)
gen_motion_tilt(mf_path, g_hold, g_pull, v_pull, v_len)
# # =======================  generate motion ===========================
if found_cnoid: 
    plan_success = load_motionfile(mf_path)
    print("plannning success? ", plan_success)
    nxt = NxtRobot(host='[::]:15005')
    motion_seq = get_motion()
    num_seq = int(len(motion_seq)/20)
    print(f"Total {num_seq} motion sequences! ")
    motion_seq = np.reshape(motion_seq, (num_seq, 20))

    # nxt.playMotionSeq(motion_seq)

# if found_cnoid: 
#     # gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra], dest="mid")
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
# main_proc_print("Time: {:.2f}s".format(end - start))
