import os
import random
import importlib
FOUND_CNOID = importlib.util.find_spec("cnoid") is not None
PLAY = True
if FOUND_CNOID: 
    from cnoid.Util import *
    from cnoid.Base import *
    from cnoid.Body import *
    from cnoid.BodyPlugin import *
    from cnoid.GraspPlugin import *
    from cnoid.BinPicking import *

from bpbot.binpicking import *
from bpbot.config import BinConfig
from bpbot.robotcon.nxt.nxtrobot_client import NxtRobot

import timeit
import numpy as np
start = timeit.default_timer()

# ---------------------- get config info -------------------------
root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
print(f"Root directory: {root_dir} ")

img_path = os.path.join(root_dir, "data/depth/depth.png")
crop_path = os.path.join(root_dir, "data/depth/cropped.png")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

config_path = os.path.join(root_dir, "config/config_cable.yaml")
cfg = BinConfig(config_path=config_path)
cfgdata = cfg.data
camera_mode = cfgdata["exp"]["camera_mode"]
log_mode = cfgdata["exp"]["log_mode"]
lr_arm = "left"

# ---------------------- get depth img -------------------------
if camera_mode:
    print("Captured point cloud ... ")
    point_array = capture_pc()
    if point_array is None:
        raise SystemExit("Capture point cloud failed ..")
    img, img_blur = pc2depth(point_array, cfgdata)
    cv2.imwrite(img_path, img_blur)
    crop = crop_roi(img_blur, cfgdata, bounding_size=50)
    cv2.imwrite(crop_path, crop)
    
# pcd = o3d.io.read_point_cloud("./data/test/out.ply")
# point_array = pcd.points

# ---------------------- compute grasps -------------------------
grasps = detect_grasp(n_grasp=10, 
                            img_path=crop_path, 
                            g_params=cfgdata['graspability'],
                            h_params=cfgdata['hand'][lr_arm])

if grasps is None:
    raise SystemError("Grasp detection failed! ")

grasp_pixels = np.array(grasps)[:, 0:2]
best_action_idx, best_grasp_idx = predict_action_grasp(grasp_pixels, crop_path)
best_grasp = grasps[best_grasp_idx]

print("Grasp  # %d | Pixel: (%d,%d,%.1f)" % (best_grasp_idx, *best_grasp,)) 
print("Action # %d" % (best_action_idx,))
img_grasp = draw_grasp(grasps, crop_path, cfgdata["hand"][lr_arm], top_only=True)
cv2.imwrite(draw_path, img_grasp)

if camera_mode:
    obj_pose, eef_pose = transform_image_to_robot(best_grasp, point_array, cfgdata, arm=lr_arm)
    print("Object 3D location: (%.3f,%.3f,%.3f)" % (*obj_pose,))
    print("Robot EEF pose (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" % (*eef_pose,)) 

    print("Generate motion file .. ")
    gen_motionfile_pick(mf_path, eef_pose, arm=lr_arm, action_idx=best_action_idx)

# ---------------------- execute on robot -------------------------
if FOUND_CNOID and camera_mode: 
    success = load_motionfile(mf_path)
    
    motion_seq = get_motion()
    
    if success and PLAY: 
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
        nxt = NxtRobot(host='[::]:15005')

        nxt.playMotion(motion_seq)
        
        if log_mode:
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y%m%d%H%M%S')
            save_dir = f"/home/hlab/Desktop/exp/{tstr}" 
            os.mkdir(save_dir)
            np.savetxt(os.path.join(save_dir, "out.txt"), np.asarray(grasps), delimiter=',')
            cv2.imwrite(os.path.join(save_dir, "grasp.png"), img_grasp)
            cv2.imwrite(os.path.join(save_dir, "depth.png"), crop)

end = timeit.default_timer()
print("[*] Time: {:.2f}s".format(end - start))
