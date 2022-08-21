import os
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

import numpy as np
import timeit

N = 1

start = timeit.default_timer()

root_dir = os.path.join(topdir, "ext/bpbot/")
#root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
main_print(f"Execute script at {root_dir} ")

img_pb_path = os.path.join(root_dir, "data/depth/depth_pick_zone.png")
img_db_path = os.path.join(root_dir, "data/depth/depth_drop_zone.png")
crop_pb_path = os.path.join(root_dir, "data/depth/depth_cropped_pick_zone.png")
crop_db_path = os.path.join(root_dir, "data/depth/depth_cropped_drop_zone.png")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

# ---------------------- get config info -------------------------

bincfg = BinConfig()
cfg = bincfg.data

def pick():
    dual_arm = False 
    success = False
    # ---------------------- get depth img --------------------------
    start_t = timeit.default_timer()
    main_print("Capture point cloud ... ")
    point_array = capture_pc()
    # if not point_array: return

    # img_pb, img_pb_blur = pc2depth(point_array, cfg['pick']['distance'], cfg['width'], cfg['height'])
    # img_db, img_db_blur = pc2depth(point_array, cfg['drop']['distance'], cfg['width'], cfg['height'])
    # point_array /= 1000

    # cv2.imwrite(img_pb_path, img_pb_blur)
    # cv2.imwrite(img_db_path, img_db_blur)

    crop_pb = crop_roi(img_pb_path, cfg['pick']['margin'], bounding=True)
    crop_db = crop_roi(img_db_path, cfg['drop']['margin'], bounding=True)

    cv2.imwrite(crop_pb_path, crop_pb)
    cv2.imwrite(crop_db_path, crop_db)

    end_t_capture = timeit.default_timer()
    print("[*] Time: ", end_t_capture -  start_t)

    ret_dropbin = pick_or_sep(img_path=crop_db_path, hand_config=cfg["hand"], bin='drop')

    if ret_dropbin is None: 
        res_pickbin = pick_or_sep(img_path=crop_pb_path, hand_config=cfg["hand"], bin='pick')
        if res_pickbin is not None: 
        
            pickorsep, g_pick = res_pickbin
            
            crop_grasp = draw_grasp(g_pick, crop_pb, cfg["hand"]["schunk"], top_color=(0,255,0), top_only=True)
            cv2.imwrite(draw_path, crop_grasp)

            _, g_r_pick = transform_image_to_robot(g_pick, point_array, cfg, 
                                                hand="left", margin="pick")
            if pickorsep == 0: 
                notice_print("Untangled! Pick zone --> goal zone!") 
                gen_motion_pickorsep(mf_path, g_r_pick, dest="goal")
            else: 
                notice_print("Tangled! Pick zone --> drop zone!") 
                gen_motion_pickorsep(mf_path, g_r_pick, dest="drop")

            notice_print("Grasp (pick) : (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_pick, *g_r_pick))
            success = True

        else: 
            warn_print("Pick bin detection failed! ")
    else:
        pickorsep, action = ret_dropbin

        if pickorsep == 0:
            notice_print("Untangled! drop zone to goal zone! ") 
            g_pick = action 
            crop_grasp = draw_grasp(g_pick, crop_db, cfg["hand"]["schunk"], top_only=True)
            cv2.imwrite(draw_path, crop_grasp)

            _, g_r_pick = transform_image_to_robot(g_pick, point_array, cfg, 
                                                hand="left", margin="drop")
            notice_print("Grasp (pick) : (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_pick, *g_r_pick))
            gen_motion_pickorsep(mf_path, g_r_pick, dest="goal")
        else: 
            # vector image to robot: swap x and y
            g_pull = action[0:3]
            g_hold = action[3:6]
            v_pull = action[6:8]
            v_r_pull = [action[7], action[6]]
            
            p_fg_pull, g_r_pull = transform_image_to_robot(g_pull, point_array, cfg, hand="left", margin="drop",dualarm=True)
            p_fg_hold, g_r_hold = transform_image_to_robot(g_hold, point_array, cfg, hand="right", margin="drop", tilt=60, dualarm=True)
            v_len = is_colliding(p_fg_pull[:2], v_pull, cfg, point_array)

            v_len = 0.1 
            notice_print("Tangle! Separation motion! ")

            notice_print("Grasp (pull): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_pull, *g_r_pull))
            notice_print("Grasp (hold): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_hold, *g_r_hold))
            notice_print("Vector (pull): (%.2f,%.2f), length: %.3f" % (*v_pull, v_len))

            # # draw grasp
            # img_grasp = draw_grasp([g_pull,g_hold], crop_db.copy())
            img_grasp = draw_hold_and_pull_grasps(g_pull, v_pull, g_hold, crop_db.copy())
            cv2.imwrite(draw_path, img_grasp)
            gen_motion_pickorsep(mf_path, g_r_pull, pose_right=g_r_hold, pulling=[*v_r_pull, v_len])
            # gen_motion_pickorsep(mf_path, g_r_pull, pulling=[*v_r_pull, v_len])
            dual_arm = True

        success = True

    if success and found_cnoid: 
        
        plan_success = load_motionfile(mf_path, dual_arm=dual_arm)
        # second motion: down the gripper
        if plan_success[1] == False: 
            warn_print(f"Approaching the target failed! ")

        if dual_arm == True and plan_success.count(True) != len(plan_success):
            warn_print(f"Dual arm planning failed! Single-arm replanning! ")
            gen_motion_pickorsep(mf_path, g_r_pull, pulling=[*v_r_pull, v_len])
            plan_success = load_motionfile(mf_path, dual_arm=False)

        main_print(f"Motion planning succeed? ==> {plan_success.count(True) == len(plan_success)}")
        
        if plan_success:
            nxt = NxtRobot(host='[::]:15005')
            motion_seq = get_motion()
            num_seq = int(len(motion_seq)/20)
            motion_seq = np.reshape(motion_seq, (num_seq, 20))
            
            nxt.playMotionSeq(motion_seq) 

        else:
            warn_print("Motion plannin failed ...")

        # tdatetime = dt.now()
        # tstr = tdatetime.strftime('%Y%m%d%H%M%S')
        # os.mkdir(f"{root_dir}/exp/{tstr}")
        # np.savetxt(f"{root_dir}/exp/{tstr}/out.txt", np.asarray(best_grasp), delimiter=',')
        # cv2.imwrite(f"{root_dir}/exp/{tstr}/grasp.png", crop_grasp)
        # import shutil
        # if res_db is not None: 
        #     # copy drop zone heatmaps 
        #     shutil.copyfile(crop_db_path, f"{root_dir}/exp/{tstr}/depth_drop_zone.png")
        #     shutil.copyfile(f"{root_dir}/data/depth/out_depth_cropped_drop_zone.png", f"{root_dir}/exp/{tstr}/out_depth_drop_zone.png")
        # else: 
        # # copy pick zone heatmaps 
        #     shutil.copyfile(crop_pb_path, f"{root_dir}/exp/{tstr}/depth_pick_zone.png")
        #     shutil.copyfile(f"{root_dir}/data/depth/out_depth_cropped_pick_zone.png", f"{root_dir}/exp/{tstr}/out_depth_pick_zone.png")

    end = timeit.default_timer()
    main_print("Time: {:.2f}s".format(end - start))

# --------------------------------------------------------------------------------
if found_cnoid:
    for i in range(N):
        pick()
else: 
    pick()
# --------------------------------------------------------------------------------