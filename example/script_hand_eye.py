import cv2
import numpy as np
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
np.set_printoptions(suppress=True)

from datetime import datetime as dt
from bpbot.robotcon.nxt.nxtrobot_client import NxtRobot
import bpbot.driver.phoxi.phoxi_client as pclt
from bpbot.utils import *
pxc = pclt.PhxClient(host="127.0.0.1:18300")

calib_mkid = 5

root_dir = "/home/hlab/bpbot"
# calib_arm = "left"
calib_arm = "right"

calib_dir = os.path.join(root_dir, "data/calibration", "20221017")
save_robot_r = os.path.join(calib_dir, "robot_clb_r.txt")
save_camera_r = os.path.join(calib_dir, "camera_clb_r.txt")
save_robot_l = os.path.join(calib_dir, "robot_clb_l.txt")
save_camera_l = os.path.join(calib_dir, "camera_clb_l.txt")

if calib_arm == "right":
    mf_path = os.path.join(root_dir, "data/motion/calib_r.dat")
    pre_robot = os.path.join(calib_dir, "robot_r.txt")
    save_robot, save_camera = save_robot_r, save_camera_r
    
elif calib_arm == "left":
    mf_path = os.path.join(root_dir, "data/motion/calib_l.dat")
    pre_robot = os.path.join(calib_dir, "robot_l.txt")
    save_robot, save_camera = save_robot_l, save_camera_l

if found_cnoid:
    camera_pos_clb = []
    robot_pos_clb = []
    robot_pos = np.loadtxt(pre_robot)
    plan_success = load_motionfile(mf_path, dual_arm=True)

    nxt = NxtRobot(host='[::]:15005')
    motion_seq = get_motion()
    num_seq = int(len(motion_seq)/20)
    print(f"[*] Total {num_seq} motion sequences! ")
    motion_seq = np.reshape(motion_seq, (num_seq, 20))
    for i, m in enumerate(motion_seq):

        nxt.setJointAngles(m[1:],tm=m[0]) # no hand open-close control
        
        pxc.triggerframe()
        gray = pxc.getgrayscaleimg()
        image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        pcd = pxc.getpcd()
        
        ids = detect_ar_marker(image.copy(), show=False)

        if i == 0:
            
            continue
        print(f"[*] Start {i:02d}-th capture! ", end="")

        # pcd_r = rotate_point_cloud(pcd)
        if calib_mkid in ids.keys(): 
            print(f"=> Detected marker {calib_mkid}")
            x, y = ids[calib_mkid]
            # camera_p = pcd_r[y*image.shape[1]+x] / 1000
            camera_p = pcd[y*image.shape[1]+x]
            camera_pos_clb.append(camera_p)
            robot_pos_clb.append(robot_pos[i-1])
        else: print(f"[!] No markers detected! ")
    camera_pos_clb = np.asarray(camera_pos_clb)
    robot_pos_clb = np.asarray(robot_pos_clb)
    
    np.savetxt(save_camera, camera_pos_clb)
    np.savetxt(save_robot, robot_pos_clb)
else:
    camera_pos_clb = np.vstack([np.loadtxt(save_camera_l), np.loadtxt(save_camera_r)])
    robot_pos_clb = np.vstack([np.loadtxt(save_robot_l), np.loadtxt(save_robot_r)])
    camera_pos_clb /= 1000
    # right arm
    robot_pos_clb[:,0] += 0.079
    robot_pos_clb[:,2] -= 0.030

    R, t = rigid_transform_3D(camera_pos_clb.T, robot_pos_clb.T)

    print(camera_pos_clb.shape, robot_pos_clb.shape)
    print("----------------------")
    H = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
    print(H)
    print("----------------------")
    np.savetxt(os.path.join(calib_dir, "calibmat.txt"), H, fmt='%.06f')

