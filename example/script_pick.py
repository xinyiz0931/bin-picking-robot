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

# ========================== define path =============================

# get root dir
#root_dir = os.path.abspath("./")
root_dir = os.path.join(topdir, "ext/bpbot")
#root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
main_proc_print(f"Start at {root_dir} ")

depth_dir = os.path.join(root_dir, "data/depth/")

# pc_path = os.path.join(root_dir, "data/pointcloud/out.ply")
img_path = os.path.join(root_dir, "data/depth/depth.png")
# img_path = os.path.join(root_dir, "data/test/depth9.png")
crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")
calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
traj_path = os.path.join(root_dir, "data/motion/motion_ik.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

cfg = BinConfig(config_path)

# ======================= get config info ============================

bincfg = BinConfig(config_path)
cfg = bincfg.data

# ======================== get depth img =============================

point_array = get_point_cloud(depth_dir, cfg['pick']['distance'],
                                 cfg['width'],cfg['height'])
point_array /= 1000
# # pcd = o3d.io.read_point_cloud("./data/test/out.ply")
#     # point_array = pcd.points

# =======================  compute grasp =============================

if point_array is not None: 
    grasps, img_input = detect_grasp_point(n_grasp=10, 
                                    img_path=img_path, 
                                    margins=cfg['pick']['margin'],
                                    g_params=cfg['graspability'],
                                    h_params=cfg["hand"]["schunk"])
    # grasps, img_input = detect_nontangle_grasp_point(n_grasp=10, 
    #                                 img_path=img_path, 
    #                                 margins=cfg["pick"]["margin"],
    #                                 g_params=cfg["graspability"], 
    #                                 h_params=cfg["hand"],
    #                                 t_params=cfg["tangle"])
    cv2.imwrite(crop_path, img_input)
else: grasps = None

# =======================  picking policy ===========================
if grasps is None:
    best_action = -1
    best_graspno = 0
    rx,ry,rz,ra = np.zeros(4)
else:

    if cfg['exp_mode'] == 0:
        # 0 -> graspaiblity
        best_grasp = grasps[0]
        best_graspno = 0
        best_action = 0 

    elif cfg['exp_mode'] == 1: 
        # 1 -> proposed circuclar picking
        grasp_pixels = np.array(grasps)[:, 1:3]
        best_action, best_graspno = predict_action_grasp(grasp_pixels, crop_path)
        best_grasp = grasps[best_graspno]

    elif cfg['exp_mode'] == 2:
        # 2 -> random circular picking
        best_grasp = grasps[0]
        best_action = random.sample(list(range(6)),1)[0]

    [best_grasp_r] = transform_image_to_robot([[best_grasp[1],best_grasp[2],best_grasp[4]]],
                    cfg["width"], calib_path, point_array, margins=cfg["pick"]["margin"])

# draw grasp 

    img_grasp = draw_grasps(grasps, crop_path,  cfg["hand"]["schunk"], top_only=True, top_idx=best_graspno, color=(73,192,236), top_color=(0,255,0))
    cv2.imwrite(draw_path, img_grasp)

    gen_success = gen_motion_pickorsep(mf_path, best_grasp_r, dest="mid")
# =======================  generate motion ===========================
if found_cnoid: 
    plan_success = load_motionfile(mf_path)
    # if gen_success and plan_success:
    if plan_success:
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
        # cv2.imwrite(f"{root_dir}/exp/{tstr}/grasp.png", img_grasp)
        # cv2.imwrite(f"{root_dir}/exp/{tstr}/depth.png", img_input)
# ======================= Record the data ===================s=========

# if success_flag:
#     tdatetime = dt.now()
#     tstr = tdatetime.strftime('%Y%m%d%H%M%S')
#     input_name = "{}_{}_{}_{}_.png".format(tstr,best_grasp[1],best_grasp[2],best_action)
#     cv2.imwrite('./exp/{}'.format(input_name), input_img)
#     cv2.imwrite('./exp/draw/{}'.format(input_name), cimg)
#     cv2.imwrite('./exp/full/{}'.format(input_name), full_image)

end = timeit.default_timer()
main_proc_print("Time: {:.2f}s".format(end - start))
