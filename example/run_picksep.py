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

start = timeit.default_timer()

# ---------------------- get config info -------------------------
root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), '../../'))
depth_dir = os.path.join(root_dir, 'data/depth')

img_path = os.path.join(depth_dir, 'depth.png')
img_buffer_path = os.path.join(depth_dir, 'depth_bufferbin.png')
crop_path = os.path.join(depth_dir, 'cropped.png')
crop_buffer_path = os.path.join(depth_dir, 'cropped_bufferbin.png')
draw_path = os.path.join(depth_dir, 'result.png')

vis_pickmain_path = os.path.join(depth_dir, 'pred/picknet_cropped.png')
vis_pickbuffer_path = os.path.join(depth_dir, 'pred/picknet_cropped_bufferbin.png')
vis_pullbuffer_path = os.path.join(depth_dir, 'pred/pullnet_cropped_bufferbin.png')
score_pick_path = os.path.join(depth_dir, 'pred/picknet_score.pickle')
score_pull_path = os.path.join(depth_dir, 'pred/pullnet_score.pickle')
vis_path = os.path.join(depth_dir, 'vis.png')

mf_path = os.path.join(root_dir, 'data/motion/motion.dat')
config_path = os.path.join(root_dir, 'config/config_picksep.yaml')

cfg = BinConfig(config_path=config_path)
cfgdata = cfg.data
exp_mode = cfgdata['exp']['mode']
mf_mode = cfgdata['exp']['motionfile_mode']
log_mode = cfgdata['exp']['log_mode']
iteration = cfgdata['exp']['iteration']

def demo():
    gen_success = False
    if log_mode:
        j = {}
        j['grasp'] = []
        j['pull'] = []
        j['container'] = ''

    # ---------------------- get depth img --------------------------
    point_array = capture_pc() # unit: m

    if point_array is not None: 
        print("Capture point cloud ... ")

        _, img_blur = pc2depth(point_array, cfgdata, container='main')
        _, img_buffer_blur = pc2depth(point_array, cfgdata, container='buffer')

        cv2.imwrite(img_path, img_blur)
        cv2.imwrite(img_buffer_path, img_buffer_blur)

        crop_main = crop_roi(img_blur, cfgdata, container='main')
        crop_buffer = crop_roi(img_buffer_blur, cfgdata, container='buffer')

        cv2.imwrite(crop_path, crop_main)
        cv2.imwrite(crop_buffer_path, crop_buffer)

    ret_bufferbin = pick_or_sep(img_path=crop_buffer_path, hand_config=cfgdata['hand'], bin='buffer')

    if ret_bufferbin is None: 
        ret_mainbin = pick_or_sep(img_path=crop_path, hand_config=cfgdata['hand'], bin='main')

        if ret_mainbin is not None: 
        
            pickorsep, g_pick = ret_mainbin
            img_grasp = draw_grasp(g_pick, crop_path, cfgdata['hand']['left'], top_color=(0,255,0), top_only=True)

            cv2.imwrite(draw_path, img_grasp)

            if point_array is not None: 
                obj_pose_pick, eef_pose_pick = transform_image_to_robot(g_pick, point_array, cfgdata, 
                                                                    arm='left', container="main")
                if pickorsep == 0: 
                    print("**Untangled**! Pick zone --> goal zone!") 
                    gen_motionfile_picksep(mf_path, eef_pose_pick, dest="side")
                else: 
                    print("**Tangled**! Pick zone --> drop zone!") 
                    gen_motionfile_picksep(mf_path, eef_pose_pick, dest="drop")

                print("**Pick**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pick, *obj_pose_pick))
                gen_success = True
            else:
                print("**Pick**! Grasp : (%d,%d,%.1f)" % (*g_pick,))
            
            heatmaps = cv2.imread(vis_pickmain_path)
            vis_buffer = cv2.imread(crop_buffer_path)
            vis_main = img_grasp
            if log_mode:
                j['grasp'] = g_pick
                j['pull'] = []
                j['container'] = 'pick'
                j['picknet_score'] = pickle.load(open(score_pick_path, 'rb'))

    else:
        pickorsep = ret_bufferbin[0]

        if pickorsep == 0:
            _, g_pick = ret_bufferbin
            print("**Untangled**! Drop zone to goal zone! ") 
            img_grasp = draw_grasp(g_pick, crop_buffer_path, cfgdata["hand"]["left"], top_only=True)
            cv2.imwrite(draw_path, img_grasp)

            if point_array is not None:
                obj_pose_pick, eef_pose_pick = transform_image_to_robot(g_pick, point_array, cfgdata, 
                                                    arm='left', container='buffer')
                print("**Pick**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pick, *obj_pose_pick))
                gen_motionfile_picksep(mf_path, eef_pose_pick, dest="side")
                gen_success = True
            else:
                print("**Pick**! Grasp : (%d,%d,%.1f)" % (*g_pick,))
            
            heatmaps  = cv2.imread(vis_pickbuffer_path)
            vis_main = cv2.imread(crop_path)
            vis_buffer = img_grasp
            if log_mode:
                j['grasp'] = g_pick
                j['pull'] = []
                j['container'] = 'drop'
                j['picknet_score'] = pickle.load(open(score_pull_path, 'rb'))

        else: 
            if exp_mode == 'pdp':
                _, g_pull,v_pull = ret_bufferbin
                
                img_grasp = draw_pull_grasps(crop_buffer_path, g_pull, v_pull)
                cv2.imwrite(draw_path, img_grasp)

                if point_array is not None:
                    # obj_pose_pull, eef_pose_pull = transform_image_to_robot(g_pull, point_array, cfgdata, arm='left', container='buffer')
                    obj_pose_pull, eef_pose_pull = transform_image_to_robot(g_pull, point_array, cfgdata, arm='left', container='buffer')
                    v_pull_wrist = [v_pull[1], v_pull[0],0.06] # swap x and y from image to robot coordinate:w

                    # v_len = is_colliding(obj_pose_pull[:2], v_pull, cfg, point_array)
                    v_len = 0.08

                    print("**Pull**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pull, *obj_pose_pull))
                    print("**Pull**! Direction: (%.2f,%.2f), distance: %.3f" % (*v_pull, v_len))
                    gen_motionfile_picksep(mf_path, eef_pose_pull, pulling=[*v_pull_wrist, v_len], dest="side")
                    gen_success = True
                else:
                    print("**Pull**! Grasp : (%d,%d,%.1f)" % (*g_pull,))
                    print("**Pull**! Direction: (%.2f,%.2f)" % (*v_pull,))

            elif exp_mode == "pd": 
                _, g_drop, _ = ret_bufferbin
                img_grasp = draw_grasp(g_drop, crop_buffer_path, cfgdata["hand"]["left"], top_only=True)
                cv2.imwrite(draw_path, img_grasp)
                if point_array is not None:
                    p_drop_tcp, g_drop_wrist = transform_image_to_robot(g_drop, point_array, cfgdata, 
                                                        hand="left", container="drop")
                    print("**Drop**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_drop, *p_drop_tcp))
                    gen_motionfile_picksep(mf_path, g_drop_wrist, dest="drop")
                    gen_success = True
                else:
                    print("**Drop**! Grasp : (%d,%d,%.1f)" % (*g_drop,))
            
            h_pick = cv2.imread(vis_pickbuffer_path)
            h_sep = cv2.resize(cv2.imread(vis_pullbuffer_path), (h_pick.shape[1], h_pick.shape[0]))
            heatmaps = cv2.vconcat([h_pick, h_sep])
            vis_main = cv2.imread(crop_path)
            vis_buffer = img_grasp
            if log_mode:
                if exp_mode == 'pdp':
                    j['grasp'] = g_pull
                    j['pull'] = v_pull
                elif exp_mode == 'pd':
                    j['grasp'] = g_drop
                    j['pull'] = []
                j['container'] = 'drop'
                j['picknet_score'] = pickle.load(open(score_pick_path, 'rb'))
                j['pullnet_score'] = pickle.load(open(score_pull_path, 'rb'))

    if ret_mainbin is not None or ret_bufferbin is not None:
        viss = []
        for v, s in zip([heatmaps, vis_main, vis_buffer], ['Predicted Heatmaps', 'Action in Main Bin', 'Action in Buffer Bin']):
            v = cv2.resize(v, (int(500*v.shape[1]/v.shape[0]),500))
            v_with_title = cv_plot_title(v, s)
            viss.append(v_with_title)
        ret = cv2.hconcat(viss)
        cv2.imwrite(vis_path,ret) 
    
    if gen_success and FOUND_CNOID: 
        
        plan_success = load_motionfile(mf_path)

        print(f"Motion planning succeed? ==> {plan_success}")
        
        if plan_success:
            nxt = NxtRobot(host='[::]:15005')
            motion_seq = get_motion()
            print(motion_seq.shape)
            # new function, not test
            nxt.playMotion(motion_seq)
            
            # old_lhand = "STANDBY"
            # old_rhand = "STANDBY"
            # for m in motion_seq:
            #     if (m[-2:] != 0).all(): lhand = "OPEN"
            #     else: lhand = "CLOSE"
            #     if (m[-4:-2] != 0).all(): rhand = "OPEN"
            #     else: rhand = "CLOSE"
            #     if old_rhand != rhand:
            #         if rhand == "OPEN": nxt.openHandToolRgt()
            #         elif rhand == "CLOSE": nxt.closeHandToolRgt()
            #     if old_lhand != lhand:
            #         if lhand == "OPEN": nxt.openHandToolLft()
            #         elif lhand == "CLOSE": nxt.closeHandToolLft()
            #     old_lhand = lhand
            #     old_rhand = rhand
            #     nxt.setJointAngles(m[3:], tm=m[0])

            # nxt.playMotion(motion_seq) 

            # ----------------------------- save log ----------------------------------
            if log_mode:
                import shutil
                tdatetime = dt.now()
                tstr = tdatetime.strftime('%Y%m%d%H%M%S')
                save_dir = '/home/hlab/Desktop/exp' 
                shutil.copytree(os.path.join(cfg.depth_dir, 'pred'), os.path.join(save_dir, tstr))
                shutil.copyfile(crop_path, os.path.join(save_dir, tstr, 'depth_pick.png'))
                shutil.copyfile(crop_buffer_path, os.path.join(save_dir, tstr, 'depth_drop.png'))
                with open(os.path.join(save_dir, tstr, 'out.pickle'), 'wb') as f:
                    pickle.dump(j, f, protocol=pickle.HIGHEST_PROTOCOL)
                cv2.imwrite(os.path.join(save_dir, tstr, 'vis.png'), ret)
            # ----------------------------- save log ----------------------------------

        else:
            print("Motion planning failed ...")


    end = timeit.default_timer()
    print("Time: {:.2f}s".format(end - start))

# --------------------------------------------------------------------------------
if FOUND_CNOID:
    for i in range(iteration):
        demo()
else: 
    demo()
# --------------------------------------------------------------------------------
