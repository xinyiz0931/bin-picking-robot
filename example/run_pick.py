import os
import random
import importlib
FOUND_CNOID = importlib.util.find_spec("cnoid") is not None
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

import timeit
import numpy as np
start = timeit.default_timer()

# ---------------------- pre-defined -------------------------
PLAY = True

CONTAINER = "pick"
ARM = "left"
FOUND_GRASP = False
PLACE_XYZ = [0.48,0.3,0.25]
if ARM == "right": PLACE_XYZ[1] *= -1
LOG_ON = False


# root_dir = os.path.join(topdir, "ext/bpbot")
root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
print(f"[*] Execute script at {root_dir} ")

img_path = os.path.join(root_dir, "data/depth/depth.png")
crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")
calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
traj_path = os.path.join(root_dir, "data/motion/motion_ik.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

# ---------------------- get config info -------------------------
cfg = BinConfig(config_path)
cfgdata = cfg.data

# ---------------------- get depth img -------------------------

point_array = capture_pc()
if point_array is not None:
    print("[*] Captured point cloud ... ")
    img, img_blur = pc2depth(point_array, cfgdata, container=CONTAINER)
    cv2.imwrite(img_path, img_blur)
    
    crop = crop_roi(img_blur, cfgdata, container=CONTAINER, bounding_size=50)

    cv2.imwrite(crop_path, crop)

# pcd = o3d.io.read_point_cloud("./data/test/out.ply")
# point_array = pcd.points

# ---------------------- compute grasps -------------------------
# h_params = {
#     "finger_length": 25,
#     "finger_width":  8, 
#     "open_width":    60
#     # "open_width":    40
# }
    
# g_params = {
#     "rotation_step": 22.5, 
#     "depth_step":    10,
#     "hand_depth":    20
# }

grasps = detect_grasp(n_grasp=10, 
                            img_path=crop_path, 
                            g_params=cfgdata['graspability'],
                            h_params=cfgdata["hand"][ARM])
                            # g_params=g_params,
                            # h_params=h_params)

# grasps = detect_nontangle_grasp(n_grasp=10, 
#                                 img_path=crop_path, 
#                                 g_params=g_params, 
#                                 h_params=h_params,
#                                 t_params=cfgdata["tangle"])


# grasps = picknet(img_path=crop_path, hand_config=cfgdata["hand"]["right"])

# ---------------------- picking policy -------------------------
if grasps is None:
    print(f"[!] No grasp detected! ")
    best_action_idx = -1
    best_grasp_wrist = 6 * [0]

else:
    # 0 -> graspaiblity
    FOUND_GRASP = True
    best_grasp = grasps[0]
    best_grasp_idx = 0
    best_action_idx = 0 
    print("test:", best_grasp)
    
    best_grasp_tcp, best_grasp_wrist = transform_image_to_robot(best_grasp, point_array, cfgdata, 
                                               hand=ARM, container=CONTAINER)

# draw grasp
    print("[*] Pick | Grasp: (%d,%d,%.1f)" % (*best_grasp,)) 
    print("[*] Pick | TCP (%.3f,%.3f,%.3f), Wrist (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                 % (*best_grasp_tcp, *best_grasp_wrist)) 
    

    # img_grasp = draw_grasp(grasps, crop.copy(),  cfgdata["hand"][ARM], top_only=True, top_idx=best_grasp_idx, color=(73,192,236), top_color=(0,255,0))
    img_grasp = draw_grasp(grasps, crop_path, cfgdata["hand"][ARM], top_only=True, top_idx=best_grasp_idx)
    # img_grasp = draw_grasp(grasps, crop_path, h_params, top_only=False, top_idx=best_grasp_idx)
    cv2.imwrite(draw_path, img_grasp)

    gen_motion_pick(mf_path, best_grasp_wrist, action_idx=best_action_idx)
# ---------------------- execute on robot -------------------------

if FOUND_CNOID and FOUND_GRASP: 
    # using motion file planner
    # success = load_motionfile(mf_path)

    #using bin picking planner: not robust 
    success = plan_binpicking(ARM, best_grasp_wrist[:3], best_grasp_wrist[3:], PLACE_XYZ )
    
    motion_seq = get_motion()
    
    if success and PLAY: 
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
        nxt = NxtRobot(host='[::]:15005')

        nxt.playMotion(motion_seq)
        
        if LOG_ON:
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y%m%d%H%M%S')
            save_dir = f"/home/hlab/Desktop/exp/{tstr}" 
            os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, "out.txt"), np.asarray(grasps), delimiter=',')
            cv2.imwrite(os.path.join(save_dir, "grasp.png"), img_grasp)
            cv2.imwrite(os.path.join(save_dir, "depth.png"), crop)

end = timeit.default_timer()
print("[*] Time: {:.2f}s".format(end - start))
