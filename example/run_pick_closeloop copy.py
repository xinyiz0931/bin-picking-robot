from cnoid.Util import *
from cnoid.Base import *
from cnoid.Body import *
from cnoid.BodyPlugin import *
from cnoid.GraspPlugin import *
from cnoid.BinPicking import *
import time
import datetime
import numpy as np
np.set_printoptions(suppress=True)
import os
from bpbot.binpicking import *
from bpbot.config import BinConfig
from bpbot.robotcon.nxt.nxtrobot_client import NxtRobot
from bpbot.device import DynPickClient, DynPickControl
from bpbot.decision_tree import DecisionTree, DecisionWindow

# global arguments
PLAY = False

CAPTURE = False

ARM = "right"

sensitivity = np.array([32.800,32.815,32.835,1653.801,1634.816,1636.136])
F0 = np.array([8198,8215,8582,8310,8078,8333])

nxt = NxtRobot(host='[::]:15005')
dc = DynPickClient()
dt = DecisionTree()
fs = DynPickControl()
# mtm = 0.1
FZ_THLD = 1
curr_jnt = []

def close_to(x, v, delta=0.01):
    if x > v-delta and x < v+delta:
        return True
    else:
        return False

if CAPTURE: 
    # initialize
    root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
    print(f"[*] Execute script at {root_dir} ")

    img_path = os.path.join(root_dir, "data/depth/depth.png")
    crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
    draw_path = os.path.join(root_dir, "data/depth/result.png")

    # ---------------------- get config info -------------------------
    cfg = BinConfig()
    cfgdata = cfg.data
    
    point_array = capture_pc()
    if point_array is not None:
        print("[*] Captured point cloud ... ")
        img, img_blur = pc2depth(point_array, cfgdata)
        cv2.imwrite(img_path, img_blur)
        
        crop = crop_roi(img_blur, cfgdata, bounding=True)

        cv2.imwrite(crop_path, crop)
        
        grasps = detect_grasp(n_grasp=5, 
                            img_path=crop_path, 
                            g_params=cfgdata['graspability'],
                            h_params=cfgdata["hand"][ARM])
        if grasps is None:
            print(f"[!] No grasp detected! ")
            raise SystemExit("[!] Failed! ")
        
        best_grasp = grasps[0]
        best_grasp_idx = 0
        best_action_idx = 0 
        
        best_grasp_tcp, best_grasp_wrist = transform_image_to_robot(best_grasp, point_array, cfgdata, 
                                                hand=ARM)
        img_grasp = draw_grasp(grasps, crop_path, cfgdata["hand"][ARM], top_only=True, top_idx=best_grasp_idx)
        print("[*] Pick | Grasp: (%d,%d,%.1f)" % (*best_grasp,)) 
        print("[*] Pick | TCP (%.3f,%.3f,%.3f), Wrist (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                    % (*best_grasp_tcp, *best_grasp_wrist)) 

        cv2.imwrite(draw_path, img_grasp)

        tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        cv2.imwrite("/home/hlab/Desktop/exp2023/"+tstr+"_img.png", crop)
        cv2.imwrite("/home/hlab/Desktop/exp2023/"+tstr+"_ret.png", img_grasp)

else: 
    best_grasp_wrist = [0.501, 0.052, 0.217, 90,-90,-90]

def pick(pose):
    global ft_reset
    
    print("[*] ------------------------------------------")
    print("[*] Picking motion planning! ")
    success = plan_pick(ARM, pose[:3], pose[3:], init=True)
    motion_seq = get_motion()
    if not success: 
        print("[!] Planning failed! ")
        raise SystemExit("[!] Failed! ")
    elif PLAY: 
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
        # nxt.playMotion(motion_seq)
        nxt.playMotion(motion_seq[:3])
        time.sleep(1) 
        _ft0 = []
        set_zero_tm = 1
        for i in np.arange(0,set_zero_tm,0.05):
            if i > 0.1:
                print(dc.get())
                _ft0.append(dc.get())
            time.sleep(0.05)
            
        global F0_reset
        
        # _ft0 = np.array(_ft0)
        # _ft0_filter = ft.filter(np.arange(_ft0.shape[0]), _ft0)[1]
        # F0_reset = np.mean(ft.filter(np.arange(_ft0.shape[0]), _ft0)[1], axis=0).astype(int)
        F0_reset = np.mean(_ft0, axis=0).astype(int)
        print("--- Predeinfed: ", F0)
        print("--- Recalib filtered force/torque: ", F0_reset)
        # print("--- Recalib force/torque: ", np.mean(_ft0, axis=0).astype(int))
        
        nxt.playMotion(motion_seq[3:])


def lift(pose):
    print("[*] ------------------------------------------")
    print("[*] Lifting motion planning! ")
    pose_after = [0.500,-0.054,0.570,90,-90,-90]
    success = plan_lift(ARM, pose[:3], pose[3:], pose_after[:3], pose_after[3:])
    motion_seq = get_motion()
    if not success: 
        print("[!] Planning failed! ")
        raise SystemExit("[!] Failed! ")
    elif PLAY:
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")

        nxt.playMotion(motion_seq, wait=False)
        curr_jnt, fout = monitoring_ft(max_tm=np.sum(motion_seq[:,0]), thld=1.75)
        tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # np.savetxt("/home/hlab/Desktop/exp2023/"+tstr+"_lift.txt",fout) 
        fs.plot(fout, filter=True)
        return curr_jnt, np.mean(fout[-10:,:],axis=0)

    
def regrasp():
    flip = False
    front_flip_pose = [0.480,-0.170,0.450,180,0,-90]
    back_flip_pose = [0.480,-0.170,0.450,0,0,-90]
    pose_after = [0.500,-0.054,0.570,-102.4,-90,102.4]
    success = plan_move(ARM, front_flip_pose[:3], front_flip_pose[3:])
    motion_seq = get_motion()
    print(motion_seq)
    if not success:
        print("[!] Planning failed!")
        raise SystemExit("[!] Failed! ")
    tmp_tz = 0
    if PLAY:
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
        nxt.playMotion(motion_seq)
        _, fout = monitoring_ft(max_tm=1, stop=False)
        # ft = np.mean(fout[:,-10:], axis=0)
        ft_front = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
        print("regrasp m1: ", ft_front)
        # if not close_to(ft[-1], 0, delta=0.03):
        # if ft[-1] > 0.03 or ft[-1] < -0.03: 
        flip = True
        print("FLIP!")
            
        print("Determine flip or not:", ft)
        fs.plot(fout, filter=True)
    
    if PLAY:
        clear_motion([])
        if flip:
            success = plan_move(ARM, back_flip_pose[:3], back_flip_pose[3:])
            motion_seq = get_motion()
            if not success:
                print("Planning failed!")
                raise SystemExit("Failed")
            if PLAY: 
                print("Planning success! ")
                nxt.playMotion(motion_seq)
                _, fout = monitoring_ft(max_tm=1, stop=False)
                ft_back = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
                print("regrasp m2: ", ft_back)

                clear_motion([])
                if np.abs(ft_back[5]) < np.abs(ft_front[5]):
                    print("Procedd!")
                    success = plan_regrasp(back_flip_pose[:3], back_flip_pose[3:], pose_after[:3], pose_after[3:])
                else:
                    print("Flip back! ")
                    success = plan_regrasp(front_flip_pose[:3], front_flip_pose[3:], pose_after[:3], pose_after[3:])

                motion_seq = get_motion()
                if not success:
                    print("[!] Planning failed!")
                    raise SystemExit("[!] Failed! ")

                if PLAY:
                    print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
                    nxt.playMotion(motion_seq)

        
    # print("[*] ------------------------------------------")
    # print("[*] Regrasping motion planning! ")
    # success = plan_regrasp(pose[:3], pose[3:])
    # motion_seq = get_motion()
    # if not success:
    #     print("[!] Planning failed!")
    #     raise SystemExit("[!] Failed! ")

    # if PLAY:
    #     print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
    #     nxt.playMotion(motion_seq)

def put_back():
    back_pose = [0.480, -0.010, 0.480, 90, -90, -90]
    success = plan_put(ARM, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
    print(f"Total {motion_seq.shape[0]} motion sequences! ")
    if not success:
        print("[!] Planning failed! ")
        raise SystemExit("[!] Failed! ")
    if PLAY: 
        print("Success! ") 
        nxt.playMotion(motion_seq)

def put_ok():
    back_pose = [-0.050, -0.550, 0.450, 90, -90, -90]
    success = plan_put(ARM, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
    print(f"Total {motion_seq.shape[0]} motion sequences! ")
    if not success:
        print("[!] Planning failed! ")
        raise SystemExit("[!] Failed! ")
    if PLAY: 
        print("[*] Success! ")
        nxt.playMotion(motion_seq)

def fling(j3, j4, vel=1):
    #TODO orientation and velocity
    print("[*] ------------------------------------------")
    print("[*] Flinging motion planning! ")
    success = plan_fling(ARM, j3=j3, j4=j4, vel=vel)
    motion_seq = get_motion()
    if not success:
        print("[!] Planning failed! ")
        raise SystemExit("[!] Failed! ")
    if PLAY: 
        print("[*] Success! ")
        nxt.playMotion(motion_seq)
    time.sleep(2)

def place():
    print("[*] ------------------------------------------")
    print("[*] Placing motion planning! ")
    success = plan_place(ARM, [],[])
    motion_seq = get_motion()
    if not success:
        print("[!] Planning failed! ")
        raise SystemExit("[!] Failed! ")
    if PLAY: 
        print("[*] Success! ")
        nxt.playMotion(motion_seq, wait=False)
        print("Time: ", motion_seq[:,0])
        curr_jnt, fout = monitoring_ft(max_tm=np.sum(motion_seq[:,0]), thld=2)
        tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # np.savetxt("/home/hlab/Desktop/exp2023/"+tstr+"_place.txt",fout) 
        fs.plot(fout, filter=True)
        return (curr_jnt, np.mean(fout[-10:,:],axis=0))
        
def monitoring_ft(max_tm=6, frequeny=20, thld=2, stop=True):
    _tm = 1/frequeny
    ft_out = []
    curr_jnt = []
    print("-----------------")
    for i in np.arange(0, max_tm, _tm):
        ft = dc.get()
        # print("@Pre: ", (ft-F0)/sensitivity)
        # print("@Pst: ", (ft-F0_reset)/sensitivity)
        # print("-----")

        ft_out.append((ft-F0_reset)/sensitivity)
        # ft_out.append((ft-F0)/sensitivity)
        # ft_out.append(ft)
        if len(ft_out) < 10:
            ft_ = np.mean(ft_out, axis=0)
        else:
            ft_ = np.mean(ft_out[-10:], axis=0)

        # fz = np.mean(ft_out, axis=0)[2]
        print("@pst & smooth: ", ft_)
        fz = ft_[2]
        if stop:
            if (fz > thld and i>max_tm/4): 
                print("Stopping the motion! Fz=%.3f [N]" % fz)
                nxt.stopMotion()
                curr_jnt = nxt.getJointAngles()
                break
        time.sleep(_tm)
    print("-----------------")
    time.sleep(1)
    return curr_jnt, np.array(ft_out)

pick(best_grasp_wrist)
clear_motion([])

ret = lift(best_grasp_wrist)

grasped = True
if ret is not None and ret != []:
    jnt, ft = ret
    if jnt != []:
        clear_motion(jnt)
        fling(j3=30, j4=40, vel=0.5)
    
    # if ft[2] < 0.5 and close_to(ft[5], 0, delta=0.05):
    # if close_to(ft[2], 0, delta=0.1): 
    #     grasped = False
        
    # elif ft[2] < 0.4 and close_to(ft[5], 0, delta=0.1):
    #     print("Regrasping! ")
    #     regrasp()
    
    # elif jnt != []:
    #     print("Flingling anyway! ")
    #     clear_motion([])
    #     fling(j3=30, j4=40, vel=0.5)

    # _, fout = monitoring_ft(2, stop=False)
    # ft = np.mean(fout[:,-10:], axis=0)
    # if ft[2] < 0.4 and close_to(ft[5], 0, delta=0.1):
    #     print("Regrasp!")
    # regrasp()
not_finish = True

while(not_finish):
    print("Monitoring and calculate next action ... ")
    _, fout = monitoring_ft(2, stop=False)
    fout_mean = dt.expectation(fout)
    act = dt.infer([[fout_mean[i] for i in [0,1,2,5]]])
    act_name = ['success', 'fling', 'regrasp']
    print(act, act_name[int(act[0])])

    app = DecisionWindow()
    app.master.mainloop()
    print("Supervised act: ", app.act)
    print("Fine tuning decision tree! ")
    dt.finetune([[fout_mean[i] for i in [0,1,2,5]]], [app.act])
    with open(dt.add_path, 'a') as ap:
        ap.write(np.append(fout_mean, app))
    # np.savetxt(dt.add_path, np.append(fout_mean, app.act))

    if app.act == 0: 
        clear_motion([])
        ret = place()
        
        if ret is not None:
            jnt, ft = ret
            clear_motion(jnt)
            if jnt != []:
                clear_motion(jnt)
                fling(j3=30, j4=40, vel=0.5)
            else:
                break
    elif app.act == 1:
        clear_motion([])
        fling(j3=30, j4=40, vel=0.5)

    elif app.act == 2:
        clear_motion([])
        regrasp()

put_ok()




# while (grasped):
#     clear_motion([])
#     ret = place()
#     if ret is not None:
#         jnt, ft = ret
#         print(ft)
#         clear_motion(jnt)
#         if jnt != []:
#             print("Entanglement detected! ft=",ft)
#             fling(j3=30, j4=40, vel=0.5)
#         elif ft[2] > 1.3:
#             print("Put entangled objects back! ")
#             put_back()
#         else:
#             break
# if grasped:
#     put_ok()
# else:
#     nxt.goInitial()
# print("[*] Finished! ")
