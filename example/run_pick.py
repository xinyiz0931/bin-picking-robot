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

import timeit
import numpy as np
start = timeit.default_timer()

# ---------------------- define path -------------------------

# get root dir
#root_dir = os.path.abspath("./")
root_dir = os.path.join(topdir, "ext/bpbot")
#root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
print(f"[*] Execute script at {root_dir} ")

img_path = os.path.join(root_dir, "data/depth/depth.png")
crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")
calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
traj_path = os.path.join(root_dir, "data/motion/motion_ik.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

# ---------------------- get config info -------------------------
bincfg = BinConfig(config_path)
cfg = bincfg.data

# ---------------------- get depth img -------------------------
bin = "pick"

print("[*] Capture point cloud ... ")
point_array = capture_pc()
print(point_array is None)
if cfg["depth_mode"] == "table":
    _dist = {"max": cfg["table_distance"], "min": cfg["table_distance"]-100}
    img, img_blur = pc2depth(point_array, _dist, cfg["width"],cfg["height"])
elif cfg["depth_mode"] == "bin":
    # img, img_blur = pc2depth(point_array, cfg[bin]["distance"], cfg["width"],cfg["height"])
    img, img_blur = px2depth(point_array, cfg)
cv2.imwrite(img_path, img_blur)
crop = crop_roi(img_path, cfg[bin]["margin"])

cv2.imwrite(crop_path, crop)

point_array /= 1000
# pcd = o3d.io.read_point_cloud("./data/test/out.ply")
# point_array = pcd.points

# ---------------------- compute grasps -------------------------
print("[*] Compute grasps... ")
grasps = detect_grasp(n_grasp=5, 
                            img_path=crop_path, 
                            g_params=cfg['graspability'],
                            h_params=cfg["hand"]["left"])

# grasps, img_input = detect_nontangle_grasp(n_grasp=10, 
#                                 img_path=img_path, 
#                                 margins=cfg["pick"]["margin"],
#                                 g_params=cfg["graspability"], 
#                                 h_params=cfg["hand"],
#                                 t_params=cfg["tangle"])

# ---------------------- picking policy -------------------------
if grasps is None:
    best_action_idx = -1
    best_grasp_wrist = 6 * [0]

else:
    if cfg['exp_mode'] == 0:
        # 0 -> graspaiblity
        best_grasp = grasps[0]
        best_grasp_idx = 0
        best_action_idx = 0 

    elif cfg['exp_mode'] == 1: 
        # 1 -> proposed circuclar picking
        grasp_pixels = np.array(grasps)[:, 0:2]
        best_action_idx, best_grasp_idx = predict_action_grasp(grasp_pixels, crop_path)
        best_grasp = grasps[best_grasp_idx]
        
    elif cfg['exp_mode'] == 2:
        # 2 -> random circular picking
        best_grasp = grasps[0]
        best_grasp_idx = 0
        best_action_idx = random.sample(list(range(6)),1)[0]
     
    best_grasp_tcp, best_grasp_wrist = transform_image_to_robot(best_grasp, point_array, cfg, 
                                               hand="left", margin=bin)

# draw grasp
    print("[$] Pick | Grasp: (%d,%d,%.1f)" % (*best_grasp,)) 
    print("[$] Pick | TCP (%.3f,%.3f,%.3f), Wrist (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                 % (*best_grasp_tcp, *best_grasp_wrist)) 

    # img_grasp = draw_grasp(grasps, crop.copy(),  cfg["hand"]["left"], top_only=True, top_idx=best_grasp_idx, color=(73,192,236), top_color=(0,255,0))

    img_grasp = draw_grasp(grasps, crop.copy(), cfg["hand"]["left"], top_only=False, top_idx=best_grasp_idx)
    cv2.imwrite(draw_path, img_grasp)

# ---------------------- execute on robot -------------------------

gen_motion_pick(mf_path, best_grasp_wrist, best_action_idx)

if found_cnoid: 
    plan_success = load_motionfile(mf_path)
    # if gen_success and plan_success:
    if plan_success.count(True) == len(plan_success):
        nxt = NxtRobot(host='[::]:15005')
        motion_seq = get_motion()
        num_seq = int(len(motion_seq)/20)
        print(f"Success! Total {num_seq} motion sequences! ")
        motion_seq = np.reshape(motion_seq, (num_seq, 20))

        nxt.playMotionSeq(motion_seq) 

# ---------------------- record data -------------------------
        # tdatetime = dt.now()
        # tstr = tdatetime.strftime('%Y%m%d%H%M%S')
        # os.mkdir(f"{root_dir}/exp/{tstr}")
        # np.savetxt(f"{root_dir}/exp/{tstr}/out.txt", np.asarray(best_grasp), delimiter=',')
        # cv2.imwrite(f"{root_dir}/exp/{tstr}/grasp.png", img_grasp)
        # cv2.imwrite(f"{root_dir}/exp/{tstr}/depth.png", img_input)

end = timeit.default_timer()
print("[*] Time: {:.2f}s".format(end - start))
