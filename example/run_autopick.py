import os
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
import pickle

N = 1
LOG_ON = False
MODE = "pnsn"
# MODE = "pn"

start = timeit.default_timer()

# ---------------------- get config info -------------------------

cfg = BinConfig()
cfgdata = cfg.data

# root_dir = os.path.join(topdir, "ext/bpbot/")
root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
print(f"[*] Execute script at {cfg.root_dir} ")

img_pb_path = os.path.join(cfg.depth_dir, "depth_pickbin.png")
img_db_path = os.path.join(cfg.depth_dir, "depth_dropbin.png")
crop_pb_path = os.path.join(cfg.depth_dir, "depth_cropped_pickbin.png")
crop_db_path = os.path.join(cfg.depth_dir, "depth_cropped_dropbin.png")
draw_path = os.path.join(cfg.depth_dir, "result.png")

vis_pp_path = os.path.join(cfg.depth_dir, "pred/picknet_depth_cropped_pickbin.png")
vis_pd_path = os.path.join(cfg.depth_dir, "pred/picknet_depth_cropped_dropbin.png")
vis_sd_path = os.path.join(cfg.depth_dir, "pred/pullnet_depth_cropped_dropbin.png")
score_pn_path = os.path.join(cfg.depth_dir, "pred/picknet_score.pickle")
score_sn_path = os.path.join(cfg.depth_dir, "pred/pullnet_score.pickle")
vis_path = os.path.join(cfg.depth_dir, "vis.png")

mf_path = cfg.motionfile_path

def pick():
    gen_success = False
    if LOG_ON:
        j = {}
        j["grasp"] = []
        j["pull"] = []
        j["container"] = ""

    # ---------------------- get depth img --------------------------
    # start_t = timeit.default_timer()
    point_array = capture_pc() # unit: m
    # end_t_capture = timeit.default_timer()
    # print("[*] Time: ", end_t_capture -  start_t)

    if point_array is not None: 
        print("[*] Capture point cloud ... ")

        img_pb, img_pb_blur = pc2depth(point_array, cfgdata, container='pick')
        img_db, img_db_blur = pc2depth(point_array, cfgdata, container='drop')

        cv2.imwrite(img_pb_path, img_pb_blur)
        cv2.imwrite(img_db_path, img_db_blur)

        crop_pb = crop_roi(img_pb_blur, cfgdata, container='pick', bounding=True)
        crop_db = crop_roi(img_db_blur, cfgdata, container='drop', bounding=True)

        cv2.imwrite(crop_pb_path, crop_pb)
        cv2.imwrite(crop_db_path, crop_db)

    ret_dropbin = pick_or_sep(img_path=crop_db_path, hand_config=cfgdata["hand"], bin='drop')
    if ret_dropbin is None: 
        res_pickbin = pick_or_sep(img_path=crop_pb_path, hand_config=cfgdata["hand"], bin='pick')

        if res_pickbin is not None: 
        
            pickorsep, g_pick = res_pickbin
            img_grasp = draw_grasp(g_pick, crop_pb_path, cfgdata["hand"]["left"], top_color=(0,255,0), top_only=True)

            cv2.imwrite(draw_path, img_grasp)

            if point_array is not None: 
                p_pick_tcp, g_pick_wrist = transform_image_to_robot(g_pick, point_array, cfgdata, 
                                                                    hand="left", container="pick")
                if pickorsep == 0: 
                    print("[*] **Untangled**! Pick zone --> goal zone!") 
                    gen_motion_picksep(mf_path, g_pick_wrist, dest="side")
                else: 
                    print("[*] **Tangled**! Pick zone --> drop zone!") 
                    gen_motion_picksep(mf_path, g_pick_wrist, dest="drop")

                print("[*] **Pick**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pick, *p_pick_tcp))
                gen_success = True
            else:
                print("[*] **Pick**! Grasp : (%d,%d,%.1f)" % (*g_pick,))
            
            heatmaps = cv2.imread(vis_pp_path)
            vis_db = cv2.imread(crop_db_path)
            vis_pb = img_grasp
            if LOG_ON:
                j["grasp"] = g_pick
                j["pull"] = []
                j["container"] = "pick"
                j["picknet_score"] = pickle.load(open(score_pn_path, 'rb'))

        else: 
            print("[!] Pick bin detection failed! ")
    else:
        pickorsep = ret_dropbin[0]

        if pickorsep == 0:
            _, g_pick = ret_dropbin
            print("[*] **Untangled**! Drop zone to goal zone! ") 
            img_grasp = draw_grasp(g_pick, crop_db_path, cfgdata["hand"]["left"], top_only=True)
            cv2.imwrite(draw_path, img_grasp)

            if point_array is not None:
                p_pick_tcp, g_pick_wrist = transform_image_to_robot(g_pick, point_array, cfgdata, 
                                                    hand="left", container="drop")
                print("[*] **Pick**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pick, *p_pick_tcp))
                gen_motion_picksep(mf_path, g_pick_wrist, dest="side")
                gen_success = True
            else:
                print("[*] **Pick**! Grasp : (%d,%d,%.1f)" % (*g_pick,))
            
            heatmaps  = cv2.imread(vis_pd_path)
            vis_pb = cv2.imread(crop_pb_path)
            vis_db = img_grasp
            if LOG_ON:
                j["grasp"] = g_pick
                j["pull"] = []
                j["container"] = "drop"
                j["picknet_score"] = pickle.load(open(score_pn_path, 'rb'))

        else: 
            if MODE == "pnsn":
                _, g_pull,v_pull = ret_dropbin
                
                img_grasp = draw_pull_grasps(crop_db_path, g_pull, v_pull)
                cv2.imwrite(draw_path, img_grasp)


                if point_array is not None:
                    p_pull_tcp, g_pull_wrist = transform_image_to_robot(g_pull, point_array, cfgdata, hand="left", container="drop",dualarm=True)
                    v_pull_wrist = [v_pull[1], v_pull[0],0.06] # swap x and y from image to robot coordinate:w

                    # v_len = is_colliding(p_pull_tcp[:2], v_pull, cfg, point_array)
                    v_len = 0.08

                    print("[*] **Pull**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pull, *p_pull_tcp))
                    print("[*] **Pull**! Direction: (%.2f,%.2f), distance: %.3f" % (*v_pull, v_len))
                    gen_motion_picksep(mf_path, g_pull_wrist, pulling=[*v_pull_wrist, v_len], dest="side")
                    gen_success = True
                else:
                    print("[*] **Pull**! Grasp : (%d,%d,%.1f)" % (*g_pull,))
                    print("[*] **Pull**! Direction: (%.2f,%.2f)" % (*v_pull,))

            elif MODE == "pn": 
                _, g_drop, _ = ret_dropbin
                img_grasp = draw_grasp(g_drop, crop_db_path, cfgdata["hand"]["left"], top_only=True)
                cv2.imwrite(draw_path, img_grasp)
                if point_array is not None:
                    p_drop_tcp, g_drop_wrist = transform_image_to_robot(g_drop, point_array, cfgdata, 
                                                        hand="left", container="drop")
                    print("[*] **Drop**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_drop, *p_drop_tcp))
                    gen_motion_picksep(mf_path, g_drop_wrist, dest="drop")
                    gen_success = True
                else:
                    print("[*] **Drop**! Grasp : (%d,%d,%.1f)" % (*g_drop,))
            
            h_pick = cv2.imread(vis_pd_path)
            h_sep = cv2.resize(cv2.imread(vis_sd_path), (h_pick.shape[1], h_pick.shape[0]))
            heatmaps = cv2.vconcat([h_pick, h_sep])
            vis_pb = cv2.imread(crop_pb_path)
            vis_db = img_grasp
            if LOG_ON:
                if MODE == "pnsn":
                    j["grasp"] = g_pull
                    j["pull"] = v_pull
                elif MODE == "pn":
                    j["grasp"] = g_drop
                    j["pull"] = []
                j["container"] = "drop"
                j["picknet_score"] = pickle.load(open(score_pn_path, 'rb'))
                j["pullnet_score"] = pickle.load(open(score_sn_path, 'rb'))

    viss = []
    for v, s in zip([heatmaps, vis_pb, vis_db], ["Predicted Heatmaps", "Action in Picking Bin", "Action in Buffer Bin"]):
        v = cv2.resize(v, (int(500*v.shape[1]/v.shape[0]),500))
        v_with_title = cv_plot_title(v, s)
        viss.append(v_with_title)
    ret = cv2.hconcat(viss)
    cv2.imwrite(vis_path,ret) 

    
    if gen_success and FOUND_CNOID: 
        
        plan_success = load_motionfile(mf_path)

        print(f"[*] Motion planning succeed? ==> {plan_success}")
        
        if plan_success:
            nxt = NxtRobot(host='[::]:15005')
            motion_seq = get_motion()
            print(motion_seq.shape)
            
            old_lhand = "STANDBY"
            old_rhand = "STANDBY"
            for m in motion_seq:
                if (m[-2:] != 0).all(): lhand = "OPEN"
                else: lhand = "CLOSE"
                if (m[-4:-2] != 0).all(): rhand = "OPEN"
                else: rhand = "CLOSE"
                if old_rhand != rhand:
                    if rhand == "OPEN": nxt.openHandToolRgt()
                    elif rhand == "CLOSE": nxt.closeHandToolRgt()
                if old_lhand != lhand:
                    if lhand == "OPEN": nxt.openHandToolLft()
                    elif lhand == "CLOSE": nxt.closeHandToolLft()
                old_lhand = lhand
                old_rhand = rhand
                nxt.setJointAngles(m[3:], tm=m[0])

            # nxt.playMotion(motion_seq) 

            # ----------------------------- save log ----------------------------------
            if LOG_ON:
                import shutil
                tdatetime = dt.now()
                tstr = tdatetime.strftime('%Y%m%d%H%M%S')
                save_dir = "/home/hlab/Desktop/exp" 
                shutil.copytree(os.path.join(cfg.depth_dir, "pred"), os.path.join(save_dir, tstr))
                shutil.copyfile(crop_pb_path, os.path.join(save_dir, tstr, "depth_pick.png"))
                shutil.copyfile(crop_db_path, os.path.join(save_dir, tstr, "depth_drop.png"))
                with open(os.path.join(save_dir, tstr, "out.pickle"), 'wb') as f:
                    pickle.dump(j, f, protocol=pickle.HIGHEST_PROTOCOL)
                cv2.imwrite(os.path.join(save_dir, tstr, "vis.png"), ret)
            # ----------------------------- save log ----------------------------------

        else:
            print("[!] Motion planning failed ...")


    end = timeit.default_timer()
    print("[*] Time: {:.2f}s".format(end - start))

# --------------------------------------------------------------------------------
if FOUND_CNOID:
    for i in range(N):
        pick()
else: 
    pick()
# --------------------------------------------------------------------------------
