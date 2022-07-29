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

# get root dir
#root_dir = os.path.abspath("./")
root_dir = os.path.join(topdir, "ext/bpbot/")
#root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
main_proc_print(f"Start at {root_dir} ")

depth_dir = os.path.join(root_dir, "data/depth/")

# pc_path = os.path.join(root_dir, "data/pointcloud/out.ply")
# img_path = os.path.join(root_dir, "data/depth/depth.png")
# img_path = os.path.join(root_dir, "data/test/depth9.png")
img_pz_path = os.path.join(root_dir, "data/depth/depth_pick_zone.png")
img_mz_path = os.path.join(root_dir, "data/depth/depth_mid_zone.png")
crop_pz_path = os.path.join(root_dir, "data/depth/depth_cropped_pick_zone.png")
crop_mz_path = os.path.join(root_dir, "data/depth/depth_cropped_mid_zone.png")
# crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")
calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
traj_path = os.path.join(root_dir, "data/motion/motion_ik.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

cfg = BinConfig(config_path)

# ---------------------- get config info -------------------------

bincfg = BinConfig(config_path)
cfg = bincfg.config

# ---------------------- get depth img --------------------------

point_array = capture_pc()

img_pz, img_pz_blur = pc2depth(point_array, cfg['pick']['distance'], cfg['width'], cfg['height'])
img_mz, img_mz_blur = pc2depth(point_array, cfg['mid']['distance'], cfg['width'], cfg['height'])

cv2.imwrite(img_pz_path, img_pz_blur)
cv2.imwrite(img_mz_path, img_mz_blur)

crop_pz = crop_roi(img_pz_path, cfg['pick']['margin'])
crop_mz = crop_roi(img_mz_path, cfg['mid']['margin'])

cv2.imwrite(crop_pz_path, crop_pz)
cv2.imwrite(crop_mz_path, crop_mz)

res_mz = pick_or_sep(img_path=crop_mz_path, h_params=cfg["hand"], bin='mid')



if res_mz is None: 
    warning_print("Middle zone is empty! ")
    res_pz = pick_or_sep(img_path=crop_pz_path, h_params=cfg["hand"], bin='pick')
    cls, best_grasp = res_pz
    grasp_for_drawn = [[None, best_grasp[0], best_grasp[1], None, best_grasp[2], None, None]]
    crop_grasp = draw_grasps(grasp_for_drawn, crop_pz_path,  cfg["hand"], top_color=(0,255,0), top_only=True)
    cv2.imwrite(draw_path, crop_grasp)

    (rx,ry,rz,ra) = transform_image_to_robot((best_grasp[0],best_grasp[1],best_grasp[2]),
                    img_pz_path, calib_path, point_array, cfg["pick"]["margin"]) 
    if cls == 0: 
        # notice_print(f"Pick to goal! Point: ({best_grasp[0]}, {best_grasp[1]}), angle: {best_grasp[2]*180/math.pi}") 
        notice_print("Untangled! Pick zone --> goal zone!") 
        gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra], dest="goal")
    else: 
        # notice_print(f"Pick to mid! Point: ({best_grasp[0]}, {best_grasp[1]}), angle: {best_grasp[2]*180/math.pi}") 
        notice_print("Tangled! Pick zone --> mid zone!") 
        gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra], dest="mid")
else:
    cls, best_grasp = res_mz
    grasp_for_drawn = [[None, best_grasp[0], best_grasp[1], None, best_grasp[2], None, None]]
    crop_grasp = draw_grasps(grasp_for_drawn, crop_mz_path,  cfg["hand"], top_color=(0,255,0), top_only=True)
    cv2.imwrite(draw_path, crop_grasp)

    (rx,ry,rz,ra) = transform_image_to_robot((best_grasp[0],best_grasp[1],best_grasp[2]),
                    img_mz_path, calib_path, point_array, cfg["mid"]["margin"])
    if cls == 0:
        # notice_print(f"Mid to goal! Point: ({best_grasp[0]}, {best_grasp[1]}), angle: {best_grasp[2]*180/math.pi}") 
        notice_print("Untangled! Mid zone to goal zone! ") 
        gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra], dest="goal")
    else: 
        # vector image to robot: swap x and y
        r_vx = best_grasp[4]
        r_vy = best_grasp[3]
        gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra], [r_vx,r_vy], motion_type='sep')
        # gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra], dest="mid")
        notice_print("Tangle! Separation motion! ")

if found_cnoid: 
    # gen_success = gen_motion_pickorsep(mf_path, [rx,ry,rz,ra])
    plan_success = load_motionfile(mf_path)
    if gen_success and plan_success:
        nxt = NxtRobot(host='[::]:15005')
        motion_seq = get_motion()
        num_seq = int(len(motion_seq)/21)
        print(f"Total {num_seq} motion sequences! ")
        motion_seq = np.reshape(motion_seq, (num_seq, 21))
        nxt.playMotionSeq(motion_seq) 

    # tdatetime = dt.now()
    # tstr = tdatetime.strftime('%Y%m%d%H%M%S')
    # os.mkdir(f"{root_dir}/exp/{tstr}")
    # np.savetxt(f"{root_dir}/exp/{tstr}/out.txt", np.asarray(best_grasp), delimiter=',')
    # cv2.imwrite(f"{root_dir}/exp/{tstr}/grasp.png", crop_grasp)
    # import shutil
    # if res_mz is not None: 
    #     # copy mid zone heatmaps 
    #     shutil.copyfile(crop_mz_path, f"{root_dir}/exp/{tstr}/depth_mid_zone.png")
    #     shutil.copyfile(f"{root_dir}/data/depth/out_depth_cropped_mid_zone.png", f"{root_dir}/exp/{tstr}/out_depth_mid_zone.png")
    # else: 
    # # copy pick zone heatmaps 
    #     shutil.copyfile(crop_pz_path, f"{root_dir}/exp/{tstr}/depth_pick_zone.png")
    #     shutil.copyfile(f"{root_dir}/data/depth/out_depth_cropped_pick_zone.png", f"{root_dir}/exp/{tstr}/out_depth_pick_zone.png")
end = timeit.default_timer()
main_proc_print("Time: {:.2f}s".format(end - start))
