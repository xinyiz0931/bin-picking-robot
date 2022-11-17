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
print(f"[*] Execute script at {root_dir} ")

img_pb_path = os.path.join(root_dir, "data/depth/depth_pick_zone.png")
img_db_path = os.path.join(root_dir, "data/depth/depth_drop_zone.png")
crop_pb_path = os.path.join(root_dir, "data/depth/depth_cropped_pick_zone.png")
crop_db_path = os.path.join(root_dir, "data/depth/depth_cropped_drop_zone.png")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

vis_pickbin_path = os.path.join(root_dir, "data/image/no_action_pickbin.jpg")
vis_dropbin_path = os.path.join(root_dir, "data/image/no_action_dropbin.jpg")

# ---------------------- get config info -------------------------

bincfg = BinConfig()
cfg = bincfg.data

def pick():
    dual_arm = False 
    success = False
    # ---------------------- get depth img --------------------------
    start_t = timeit.default_timer()
    print("[*] Capture point cloud ... ")
    point_array = capture_pc()
    # if not point_array: return

    # img_pb, img_pb_blur = pc2depth(point_array, cfg['pick']['distance'], cfg['width'], cfg['height'])
    # img_db, img_db_blur = pc2depth(point_array, cfg['drop']['distance'], cfg['width'], cfg['height'])
    # img_pb, img_pb_blur = px2depth(point_array, cfg, min_=-0.017)
    img_pb, img_pb_blur = px2depth(point_array, cfg)
    img_db, img_db_blur = px2depth(point_array, cfg, min_=0.010)
    point_array /= 1000

    cv2.imwrite(img_pb_path, img_pb_blur)
    cv2.imwrite(img_db_path, img_db_blur)

    crop_pb = crop_roi(img_pb_path, cfg['pick']['margin'], bounding=True)
    crop_db = crop_roi(img_db_path, cfg['drop']['margin'], bounding=True)

    cv2.imwrite(crop_pb_path, crop_pb)
    cv2.imwrite(crop_db_path, crop_db)

    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y%m%d%H%M%S')
    cv2.imwrite(os.path.join("/home/hlab/Desktop/collected", tstr+".png"), crop_db)

    end_t_capture = timeit.default_timer()
    print("[*] Time: ", end_t_capture -  start_t)

    ret_dropbin = pick_or_sep(img_path=crop_db_path, hand_config=cfg["hand"], bin='drop')

    if ret_dropbin is None: 
        res_pickbin = pick_or_sep(img_path=crop_pb_path, hand_config=cfg["hand"], bin='pick')
        if res_pickbin is not None: 
        
            pickorsep, g_pick = res_pickbin
            
            img_grasp = draw_grasp(g_pick, crop_pb, cfg["hand"]["left"], top_color=(0,255,0), top_only=True)
            cv2.imwrite(draw_path, img_grasp)

            p_pick_tcp, g_pick_wrist = transform_image_to_robot(g_pick, point_array, cfg, 
                                                hand="left", margin="pick")
            if pickorsep == 0: 
                print("[$] Untangled! Pick zone --> goal zone!") 
                gen_motion_pickorsep(mf_path, g_pick_wrist, dest="goal")
            else: 
                print("[$] Tangled! Pick zone --> drop zone!") 
                gen_motion_pickorsep(mf_path, g_pick_wrist, dest="drop")

            print("[$] Grasp (pick) : (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_pick, *g_pick_wrist))
            success = True
            # TODO: just for visualization
            heatmap_pickbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/out_depth_cropped_pick_zone.png"))
            grasp_pickbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/ret_depth_cropped_pick_zone.png"))
            vis = []
            for v in [heatmap_pickbin, cv2.hconcat([grasp_pickbin, img_grasp])]:
                vis.append(cv2.resize(v, (1000, int(v.shape[0]*1000/v.shape[1]))))
            vis_pickbin = cv2.vconcat(vis)
            vis_dropbin = (np.ones([*vis_pickbin.shape])*255).astype(np.uint8)
            cv2.putText(vis_dropbin, "Bin (Drop)",(20,550), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
            cv2.putText(vis_dropbin, "No Action",(20,700), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
            vis = cv2.hconcat([vis_pickbin, vis_dropbin])
            cv2.imwrite(os.path.join(root_dir, "data/depth/vis.png"), vis)

        else: 
            print("[!] Pick bin detection failed! ")
    else:
        pickorsep, action = ret_dropbin

        if pickorsep == 0:
            print("[$] Untangled! drop zone to goal zone! ") 
            g_pick = action 
            img_grasp = draw_grasp(g_pick, crop_db, cfg["hand"]["left"], top_only=True)
            cv2.imwrite(draw_path, img_grasp)

            p_pick_tcp, g_pick_wrist = transform_image_to_robot(g_pick, point_array, cfg, 
                                                hand="left", margin="drop")
            print("[$] Grasp (pick) : (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_pick, *g_pick_wrist))
            gen_motion_pickorsep(mf_path, g_pick_wrist, dest="goal")
        else: 
            # vector image to robot: swap x and y
            g_pull = action[0:3]
            g_hold = action[3:6]
            v_pull = action[6:8]
            v_r_pull = [action[7], action[6]]
            
            p_tcp_pull, g_wrist_pull = transform_image_to_robot(g_pull, point_array, cfg, hand="left", margin="drop",dualarm=True)
            p_tcp_hold, g_wrist_hold = transform_image_to_robot(g_hold, point_array, cfg, hand="right", margin="drop", tilt=60, dualarm=True)
            v_len = is_colliding(p_tcp_pull[:2], v_pull, cfg, point_array)

            v_len = 0.1 
            print("[$] Tangle! Separation motion! ")

            print("[$] Grasp (pull): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_pull, *g_r_pull))
            print("[$] Grasp (hold): (%d,%d,%.1f) -> joint (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                        % (*g_hold, *g_r_hold))
            print("[$] Vector (pull): (%.2f,%.2f), length: %.3f" % (*v_pull, v_len))

            # # draw grasp
            # img_grasp = draw_grasp([g_pull,g_hold], crop_db.copy())
            img_grasp = draw_hold_and_pull_grasps(crop_db.copy(), g_pull, v_pull, g_hold)

            cv2.imwrite(draw_path, img_grasp)
            # single-arm: pull
            gen_motion_pickorsep(mf_path, g_wrist_pull, pose_rgt=g_wrist_hold, pulling=[*v_r_pull, v_len], dest="side")
            # dual-arm: hold-and-pull
            gen_motion_pickorsep(mf_path, g_wrist_pull, pulling=[*v_r_pull, v_len], dest="side")
            # dual_arm = True

        success = True
        
        # TODO: just for visualization
        heatmap_dropbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/out_depth_cropped_drop_zone.png"))
        grasp_dropbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/ret_depth_cropped_drop_zone.png"))
        vis = []
        for v in [heatmap_dropbin, cv2.hconcat([grasp_dropbin, img_grasp])]:
            vis.append(cv2.resize(v, (1000, int(v.shape[0]*1000/v.shape[1]))))
        vis_dropbin = cv2.vconcat(vis)
        vis_pickbin = (np.ones([*vis_dropbin.shape])*255).astype(np.uint8)
        cv2.putText(vis_pickbin, "Bin (Pick)",(20,550), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
        cv2.putText(vis_pickbin, "No Action",(20,700), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
        vis = cv2.hconcat([vis_pickbin, vis_dropbin])
        print(vis.shape)
        cv2.imwrite(os.path.join(root_dir, "data/depth/vis.png"), vis)
        # cv2.imwrite(os.path.join(root_dir, "data/image/vis_pickbin.png"), empty_pickbin)
        # cv2.imwrite(os.path.join(root_dir, "data/image/vis_dropbin.png"), vis_dropbin)
        

    if success and found_cnoid: 
        
        plan_success = load_motionfile(mf_path, dual_arm=dual_arm)
        # second motion: down the gripper
        if plan_success[1] == False: 
            print(f"[!] Approaching the target failed! ")

        if dual_arm == True and plan_success.count(True) != len(plan_success):
            print(f"[!] Dual arm planning failed! Single-arm replanning! ")
            gen_motion_pickorsep(mf_path, g_r_pull, pulling=[*v_r_pull, v_len])
            gen_motion_pickorsep(mf_path, g_wrist_pull, pulling=[*v_r_pull, v_len])
            plan_success = load_motionfile(mf_path, dual_arm=False)

        print(f"[*] Motion planning succeed? ==> {plan_success.count(True) == len(plan_success)}")
        
        if plan_success:
            nxt = NxtRobot(host='[::]:15005')
            motion_seq = get_motion()
            num_seq = int(len(motion_seq)/20)
            motion_seq = np.reshape(motion_seq, (num_seq, 20))
            
            nxt.playMotionSeq(motion_seq) 

        else:
            print("[!] Motion plannin failed ...")

        # tdatetime = dt.now()
        # tstr = tdatetime.strftime('%Y%m%d%H%M%S')
        # os.mkdir(f"{root_dir}/exp/{tstr}")
        # np.savetxt(f"{root_dir}/exp/{tstr}/out.txt", np.asarray(best_grasp), delimiter=',')
        # cv2.imwrite(f"{root_dir}/exp/{tstr}/grasp.png", img_grasp)
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
    print("[*] Time: {:.2f}s".format(end - start))

# --------------------------------------------------------------------------------
if found_cnoid:
    for i in range(N):
        pick()
else: 
    pick()
# --------------------------------------------------------------------------------
