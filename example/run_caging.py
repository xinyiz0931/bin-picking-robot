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
img, img_blur = pc2depth(point_array, cfgdata["pick"]["height"], cfgdata["width"], cfgdata["height"])
point_array /= 1000
cv2.imwrite(img_path, img_blur)
# import open3d as o3d
# pcd = o3d.io.read_point_cloud("/home/hlab/Desktop/test_ply.ply")
# point_array = pcd.points
img_input = crop_roi(img_path, margins=cfgdata["pick"]["area"])
cv2.imwrite(crop_path, img_input)

# =======================  compute grasp =============================

drawn = img_input.copy()
p_clicked = [] # 2x2, [[pull_x, pull_y], [hold-x, hold_y]]
def on_click(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        cv2.circle(drawn,(x,y),5,(0,255,0),-1)
        # print(f"{x},{y}")
        p_clicked.append([x,y])

cv2.namedWindow('click twice to select hold and pull')
cv2.setMouseCallback('click twice to select hold and pull',on_click)
while(len(p_clicked)<2):
    cv2.imshow('click twice to select hold and pull',drawn)
    k = cv2.waitKey(20) & 0xFF
    if k == 27 or k==ord('q'):
        break
cv2.destroyAllWindows()

#p_clicked = [[102,107], [308, 243]]

p_hold, p_pull = np.array(p_clicked)

# =======================  picking policy ===========================

right_attr = [cfgdata["hand"]["right"].get(k) for k in ["finger_width", "finger_length", "open_width"]]
left_attr = [cfgdata["hand"]["left"].get(k) for k in ["finger_width", "finger_length", "open_width"]]

gripper_right = Gripper(*right_attr)
gripper_left = Gripper(*left_attr)
# theta_pull = gripper_left.point_oriented_grasp(img_input, [p_pull[0], p_pull[1]])
theta_pull = 90
theta_pull = 0
g_pull = [*p_pull, theta_pull]

theta_hold = gripper_left.point_oriented_grasp(img_input, [p_hold[0], p_pull[1]])

theta_hold = 90
g_hold = [*p_hold, theta_hold]
crop_grasp = gripper_left.draw_grasp(g_pull, img_input.copy())
crop_grasp = gripper_right.draw_grasp(g_hold, crop_grasp)

cv2.imshow("", crop_grasp)
cv2.waitKey(0)
cv2.destroyAllWindows()

# ---------------------- coordinate transformation ---------------------
p_r_pull, g_wrist_pull = transform_image_to_robot([*p_pull, theta_pull], point_array, cfg, hand="left", margin="pick",dualarm=True)
p_r_hold, g_wrist_hold = transform_image_to_robot([*p_hold, theta_hold], point_array, cfg, hand="right", tilt=60, margin="pick", dualarm=True)

#g_wrist_hold[2] -= 0.01
v_pull = ((p_r_pull-p_r_hold) / np.linalg.norm(p_r_pull-p_r_hold))[:2]

#g_wrist_hold = [0.354,-0.213,0.055,0.0,-60.0,-90.0]
#g_wrist_pull = [0.425,-0.025,0.076,0.0,-90.0,90.0]
v_pull = [0.533, 0.846]
v_len = 0.2

print("[*] Joint (hold): (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" % (*g_r_hold,))
print("[*] Joint (pull): (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" % (*g_r_pull,))

#print("[*] Grasp (pull): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                #% (*g_pull, *g_r_pull))
#print("[*] Grasp (hold): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                #% (*g_hold, *g_r_hold))
print("[*] Vector (pull): (%.3f,%.3f), length: %.3f" % (*v_pull, v_len))

gen_motion_test(mf_path, g_wrist_pull, pose_rgt=g_wrist_hold, pulling=[*v_pull,v_len])
# gen_motion_pickorsep(mf_path, g_wrist_pull, pose_rgt=g_wrist_hold, pulling=[*v_pull,v_len])
# # =======================  generate motion ===========================
if FOUND_CNOID: 
    plan_success = load_motionfile(mf_path, dual_arm=True)
    print("plannning success? ", plan_success)
    if plan_success.count(True) == len(plan_success):
        nxt = NxtRobot(host='[::]:15005')
        motion_seq = get_motion()
        num_seq = int(len(motion_seq)/20)
        print(f"[*] Total {num_seq} motion sequences! ")
        motion_seq = np.reshape(motion_seq, (num_seq, 20))

        nxt.playMotion(motion_seq)

# end = timeit.default_timer()
# print("[*] Time: {:.2f}s".format(end - start))
