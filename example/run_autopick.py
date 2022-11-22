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

# root_dir = os.path.join(topdir, "ext/bpbot/")
root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
print(f"[*] Execute script at {root_dir} ")

img_pb_path = os.path.join(root_dir, "data/depth/depth_pickbin.png")
img_db_path = os.path.join(root_dir, "data/depth/depth_dropbin.png")
crop_pb_path = os.path.join(root_dir, "data/depth/depth_cropped_pickbin.png")
crop_db_path = os.path.join(root_dir, "data/depth/depth_cropped_dropbin.png")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

vis_pickbin_path = os.path.join(root_dir, "data/image/no_action_pickbin.jpg")
vis_dropbin_path = os.path.join(root_dir, "data/image/no_action_dropbin.jpg")
vis_pp_path = os.path.join(root_dir, "data/depth/pred/picknet_depth_cropped_pickbin.png")
vis_pd_path = os.path.join(root_dir, "data/depth/pred/picknet_depth_cropped_dropbin.png")
vis_sd_path = os.path.join(root_dir, "data/depth/pred/sepnet_depth_cropped_dropbin.png")
vis_size = 1000
_hs = int(vis_size/2)
# ---------------------- get config info -------------------------

bincfg = BinConfig()
cfg = bincfg.data

def pick():
    gen_success = False
    # ---------------------- get depth img --------------------------
    start_t = timeit.default_timer()
    point_array = capture_pc()
    # end_t_capture = timeit.default_timer()
    # print("[*] Time: ", end_t_capture -  start_t)

    if point_array is not None: 
        print("[*] Capture point cloud ... ")

        img_pb, img_pb_blur = px2depth(point_array, cfg)
        img_db, img_db_blur = px2depth(point_array, cfg, min_=0.010)
        point_array /= 1000

        cv2.imwrite(img_pb_path, img_pb_blur)
        cv2.imwrite(img_db_path, img_db_blur)

        crop_pb = crop_roi(img_pb_path, cfg['pick']['margin'], bounding=True)
        crop_db = crop_roi(img_db_path, cfg['drop']['margin'], bounding=True)

        cv2.imwrite(crop_pb_path, crop_pb)
        cv2.imwrite(crop_db_path, crop_db)

    #################################################################
    # Revise `crop_db_path` or `crop_pb_path` to test
    #################################################################
    # temporal collect data ....
    tdatetime = dt.now()
    tstr = tdatetime.strftime('%Y%m%d%H%M%S')
    # cv2.imwrite(os.path.join("/home/hlab/Desktop/collected", tstr+".png"), crop_db)

    ret_dropbin = pick_or_sep(img_path=crop_db_path, hand_config=cfg["hand"], bin='drop')

    if ret_dropbin is None: 
        res_pickbin = pick_or_sep(img_path=crop_pb_path, hand_config=cfg["hand"], bin='pick')

        if res_pickbin is not None: 
        
            pickorsep, g_pick = res_pickbin
            img_grasp = draw_grasp(g_pick, crop_pb_path, cfg["hand"]["left"], top_color=(0,255,0), top_only=True)

            cv2.imwrite(draw_path, img_grasp)

            if point_array is not None: 
                p_pick_tcp, g_pick_wrist = transform_image_to_robot(g_pick, point_array, cfg, 
                                                                    hand="left", margin="pick")
                if pickorsep == 0: 
                    print("[$] **Untangled**! Pick zone --> goal zone!") 
                    gen_motion_picksep(mf_path, g_pick_wrist, dest="side")
                else: 
                    print("[$] **Tangled**! Pick zone --> drop zone!") 
                    gen_motion_picksep(mf_path, g_pick_wrist, dest="drop")

                print("[$] **Pick**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pick, *p_pick_tcp))
                gen_success = True
            else:
                print("[$] **Pick**! Grasp : (%d,%d,%.1f)" % (*g_pick,))

            # TODO visualization 
            # "picknet"o + "pickbin"
            upper = cv2.resize(cv2.imread(vis_pp_path), (vis_size, _hs))
            lower = cv2.hconcat([np.zeros(_hs,_hs,3).astype(np.uint8), cv2.resize(img_grasp,(_hs,_hs))])
            print(upper.shape, lower.shape)
            vis = cv2.vconcat([upper, lower])


            # heatmap_pickbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/out_depth_cropped_pickbin.png"))
            # grasp_pickbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/ret_depth_cropped_pickbin.png"))
            # vis = []
            # for v in [heatmap_pickbin, cv2.hconcat([grasp_pickbin, img_grasp])]:
            #     vis.append(cv2.resize(v, (1000, int(v.shape[0]*1000/v.shape[1]))))
            # vis_pickbin = cv2.vconcat(vis)
            # vis_dropbin = (np.ones([*vis_pickbin.shape])*255).astype(np.uint8)
            # cv2.putText(vis_dropbin, "Bin (Drop)",(20,550), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
            # cv2.putText(vis_dropbin, "No Action",(20,700), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
            # vis = cv2.hconcat([vis_pickbin, vis_dropbin])
            # cv2.imwrite(os.path.join(root_dir, "data/depth/vis.png"), vis)

        else: 
            print("[!] Pick bin detection failed! ")
    else:
        pickorsep = ret_dropbin[0]

        if pickorsep == 0:
            _, g_pick = ret_dropbin
            print("[$] **Untangled**! Drop zone to goal zone! ") 
            # img_grasp = draw_grasp(g_pick, crop_db, cfg["hand"]["left"], top_only=True)
            img_grasp = draw_grasp(g_pick, crop_db_path, cfg["hand"]["left"], top_only=True)
            cv2.imwrite(draw_path, img_grasp)

            if point_array is not None:
                p_pick_tcp, g_pick_wrist = transform_image_to_robot(g_pick, point_array, cfg, 
                                                    hand="left", margin="drop")
                print("[$] **Pick**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pick, *p_pick_tcp))
                gen_motion_picksep(mf_path, g_pick_wrist, dest="side")
                gen_success = True
            else:
                print("[$] **Pick**! Grasp : (%d,%d,%.1f)" % (*g_pick,))
            
            # TODO visualization
            # dropbin + picknet
            upper = cv2.resize(cv2.imread(vis_pd_path), (vis_size, _hs))
            lower = cv2.hconcat([np.zeros(_hs,_hs,3).astype(np.uint8), cv2.resize(img_grasp,(_hs,_hs))])
            print(upper.shape, lower.shape)
            vis = cv2.vconcat([upper, lower])

        else: 
            _, g_pull, v_pull = ret_dropbin
            
            img_grasp = draw_pull_grasps(crop_db_path, g_pull, v_pull)
            cv2.imwrite(draw_path, img_grasp)

            if point_array is not None:
                p_pull_tcp, g_pull_wrist = transform_image_to_robot(g_pull, point_array, cfg, hand="left", margin="drop",dualarm=True)
                v_pull_wrist = [v_pull[1], v_pull[0],0] # swap x and y from image to robot coordinate
                v_len = is_colliding(p_pull_tcp[:2], v_pull, cfg, point_array)
                v_len = 0.1

                print("[$] **Pull**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pull, *p_pull_tcp))
                print("[$] **Pull**! Direction: (%.2f,%.2f), distance: %.3f" % (*v_pull, v_len))
                gen_motion_picksep(mf_path, g_pull_wrist, pulling=[*v_pull_wrist, v_len], dest="side")
                gen_success = True
            else:
                print("[$] **Pull**! Grasp : (%d,%d,%.1f)" % (*g_pull,))
                print("[$] **Pull**! Direction: (%.2f,%.2f)" % (*v_pull,))
            
            # TODO visualization
            # dropbin + sepnet
            h_pick = cv2.imread(vis_pd_path)
            h_sep = cv2.resize(cv2.imread(vis_sd_path), (h_pick.shape[1], h_pick.shape[0]))
            heatmaps = cv2.vconcat([h_pick, h_sep])
            
            img_ = cv2.resize(cv2.imread(crop_pb_path), (img_grasp.shape[1], img_grasp.shape[0]))
            rets = cv2.hconcat([img_, img_grasp])
            
            fig = plt.figure(1, figsize=(16, 6))
            ax1 = fig.add_subplot(121)
            ax1.imshow(h)
            ax2 = fig.add_subplot(122)
            ax2.imshow(ret)
            plt.show()


        # import shutil
        # shutil.copyfile("/home/hlab/bpbot/data/depth/pred/out_depth_cropped_dropbin.png", f"/home/hlab/bpbot/data/depth/{tstr}.png")
        # # TODO: just for visualization
        # heatmap_dropbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/out_depth_cropped_dropbin.png"))
        # grasp_dropbin = cv2.imread(os.path.join(root_dir, "data/depth/pred/ret_depth_cropped_dropbin.png"))
        # vis = []
        # for v in [heatmap_dropbin, cv2.hconcat([grasp_dropbin, img_grasp])]:
        #     vis.append(cv2.resize(v, (1000, int(v.shape[0]*1000/v.shape[1]))))
        # vis_dropbin = cv2.vconcat(vis)
        # vis_pickbin = (np.ones([*vis_dropbin.shape])*255).astype(np.uint8)
        # cv2.putText(vis_pickbin, "Bin (Pick)",(20,550), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
        # cv2.putText(vis_pickbin, "No Action",(20,700), cv2.FONT_HERSHEY_SIMPLEX, 5, (192,192,192), 3)
        # vis = cv2.hconcat([vis_pickbin, vis_dropbin])
        # cv2.imwrite(os.path.join(root_dir, "data/depth/vis.png"), vis)
        # cv2.imwrite(os.path.join(root_dir, "data/image/vis_pickbin.png"), empty_pickbin)
        # cv2.imwrite(os.path.join(root_dir, "data/image/vis_dropbin.png"), vis_dropbin)
        

    if gen_success and found_cnoid: 
        
        plan_success = load_motionfile(mf_path, dual_arm=False)
        # second motion: down the gripper
        if plan_success[1] == False: 
            print(f"[!] Approaching the target failed! ")

        # if dual_arm == True and plan_success.count(True) != len(plan_success):
        #     print(f"[!] Dual arm planning failed! Single-arm replanning! ")
        #     gen_motion_pickorsep(mf_path, g_r_pull, pulling=[*v_r_pull, v_len])
        #     gen_motion_pickorsep(mf_path, g_wrist_pull, pulling=[*v_r_pull, v_len])
        #     plan_success = load_motionfile(mf_path, dual_arm=False)

        print(f"[*] Motion planning succeed? ==> {plan_success.count(True) == len(plan_success)}")
        
        if plan_success.count(True) == len(plan_success):
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
        #     shutil.copyfile(crop_db_path, f"{root_dir}/exp/{tstr}/depth_dropbin.png")
        #     shutil.copyfile(f"{root_dir}/data/depth/out_depth_cropped_dropbin.png", f"{root_dir}/exp/{tstr}/out_depth_dropbin.png")
        # else: 
        # # copy pick zone heatmaps 
        #     shutil.copyfile(crop_pb_path, f"{root_dir}/exp/{tstr}/depth_pickbin.png")
        #     shutil.copyfile(f"{root_dir}/data/depth/out_depth_cropped_pickbin.png", f"{root_dir}/exp/{tstr}/out_depth_pickbin.png")

    end = timeit.default_timer()
    print("[*] Time: {:.2f}s".format(end - start))

# --------------------------------------------------------------------------------
if found_cnoid:
    for i in range(N):
        pick()
else: 
    pick()
# --------------------------------------------------------------------------------
