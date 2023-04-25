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
from bpbot.decision_tree import DecisionTree

# global arguments
PLAY = True

CAPTURE = True

ARM = "right"

sensitivity = np.array([32.800,32.815,32.835,1653.801,1634.816,1636.136])



nxt = NxtRobot(host='[::]:15005')
dc = DynPickClient()
fs = DynPickControl()
dt = DecisionTree()
# mtm = 0.1
curr_jnt = []
# for tuning
THLD_A = 0.2
THLD_B = 0.7
# THLD_A = 0.5 # for lifting one 
# THLD_B = 1.2 # for entanglement detection
THLD_C = 2 # for regrasping 
K1 = 0.2

def range_in(x, lower, upper):
    return True if x < upper and x > lower else False


def close_to(x, v, delta=0.01):
    return True if x > v-delta and x < v+delta else False

def calc_gradient(l):
    g = []
    size = len(l)
    for i in range(size-1):
        # g.append(l[size-1-i]-l[size-2-i])
        g.append(l[i+1]-l[i])
    return np.array(g)

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
    best_grasp_wrist = [0.501, 0.052, 0.227, 90,-90,-90]

def pick(pose):
    global ft_reset
    print("[*] ------ Pick ------")
    
    success = plan_pick(ARM, pose[:3], pose[3:], init=True)
    motion_seq = get_motion()
    if not success: 
        raise SystemExit("[!] Failed! ")
    if PLAY: 
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
        # nxt.playMotion(motion_seq)
        nxt.playMotion(motion_seq[:3])
        time.sleep(1) 
        _ft0 = []
        set_zero_tm = 1
        for i in np.arange(0,set_zero_tm,0.05):
            if i > 0.1:
                _ft0.append(dc.get())
            time.sleep(0.05)
            
        global F0_reset
        
        F0_reset = np.mean(_ft0, axis=0).astype(int)
        print("--- recalib F0:", F0_reset)

        nxt.playMotion(motion_seq[3:])


def lift(pose):
    print("[*] ------ Lift ------")
    pose_after = [0.500,-0.054,0.570,90,-90,-90]
    success = plan_lift(ARM, pose[:3], pose[3:], pose_after[:3], pose_after[3:])
    motion_seq = get_motion()
    
    if not success: 
        raise SystemExit("[!] Failed! ")
    elif PLAY: 
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
        time.sleep(2)
        nxt.playMotion(motion_seq, wait=False)
        curr_jnt, fout = monitoring(max_tm=np.sum(motion_seq[:,0]), thld=THLD_C)
        tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        np.savetxt("/home/hlab/Desktop/"+tstr+"_lift.txt",fout) 
        attr = fs.fit(fout[:,2])

        # fs.plot(fout, filter=True)
        return curr_jnt, np.mean(fout[-10:,:],axis=0), attr[-1]
    else:
        return [],[]

    
def regrasp():
    print("[*] ------ Lift ------")
    flip = False
    front_flip_pose = [0.480,-0.170,0.450,180,0,-90]
    back_flip_pose = [0.480,-0.170,0.450,0,0,-90]
    pose_after = [0.500,-0.054,0.570,-102.4,-90,102.4]
    success = plan_move(ARM, front_flip_pose[:3], front_flip_pose[3:])
    motion_seq = get_motion()
    if not success:
        raise SystemExit("[!] Failed! ")
    elif PLAY:
        print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
        nxt.playMotion(motion_seq)
        _, fout = monitoring(max_tm=1, stop=False)
        # ft = np.mean(fout[:,-10:], axis=0)
        ft_front = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
        print("regrasp m1: ", ft_front)
        # if not close_to(ft[-1], 0, delta=0.03):
        # if ft[-1] > 0.03 or ft[-1] < -0.03: 
        flip = True
        print("FLIP!")
            
        print("Determine flip or not:", ft)
        fs.plot(fout, filter=True)
    
        clear_motion([])
        
        if flip:
            success = plan_move(ARM, back_flip_pose[:3], back_flip_pose[3:])
            motion_seq = get_motion()
            if not success:
                raise SystemExit("[!] Failed")
            elif PLAY: 
                print(f"[*] Success! Total {motion_seq.shape[0]} motion sequences! ")
                nxt.playMotion(motion_seq)
                _, fout = monitoring(max_tm=1, stop=False)
                ft_back = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
                # print("regrasp m2: ", ft_back)

                clear_motion([])
                if np.abs(ft_back[5]) < np.abs(ft_front[5]):
                    print("Proceed!")
                    success = plan_regrasp(back_flip_pose[:3], back_flip_pose[3:], pose_after[:3], pose_after[3:])
                else:
                    print("Flip back! ")
                    success = plan_regrasp(front_flip_pose[:3], front_flip_pose[3:], pose_after[:3], pose_after[3:])

                motion_seq = get_motion()
                if not success:
                    raise SystemExit("[!] Failed! ")

                elif PLAY:
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
    print("[*] ------ Put  ------")
    back_pose = [0.480, -0.010, 0.480, 90, -90, -90]
    success = plan_put(ARM, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
    if not success:
        raise SystemExit("[!] Failed! ")
    elif PLAY: 
        print(f"[*] Total {motion_seq.shape[0]} motion sequences! ")
        nxt.playMotion(motion_seq)

def put_ok():
    print("[*] ------ Put  ------")
    back_pose = [-0.050, -0.550, 0.450, 90, -90, -90]
    success = plan_put(ARM, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
    if not success:
        raise SystemExit("[!] Failed! ")
    elif PLAY: 
        print(f"[*] Total {motion_seq.shape[0]} motion sequences! ")
        nxt.playMotion(motion_seq)

def fling(j3, j4, vel=1):
    #TODO orientation and velocity
    success = plan_fling(ARM, j3=j3, j4=j4, vel=vel)
    motion_seq = get_motion()
    if not success:
        raise SystemExit("[!] Failed! ")
    elif PLAY: 
        print(f"[*] Total {motion_seq.shape[0]} motion sequences! ")
        nxt.playMotion(motion_seq[:1])
        time.sleep(1)
        _, fout = monitoring(1, stop=False)
        ft = np.mean(fout[:,-10:], axis=0)
        print("Before fling ground truth", ft)
        nxt.playMotion(motion_seq[1:])
    time.sleep(2)

def place():
    print("[*] ------ place -----")
    success = plan_place(ARM, [],[])
    motion_seq = get_motion()
    if not success:
        raise SystemExit("[!] Failed! ")
    elif PLAY: 
        print(f"[*] Total {motion_seq.shape[0]} motion sequences! ")
        nxt.playMotion(motion_seq, wait=False)

        curr_jnt, fout = monitoring(max_tm=np.sum(motion_seq[:,0]), thld=THLD_C)
        # tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # np.savetxt("/home/hlab/Desktop/exp2023/"+tstr+"_place.txt",fout) 
        fs.plot(fout, filter=True)
        return curr_jnt, np.mean(fout[-10:,:],axis=0)
    else:
        return [], []

def monitoring(max_tm=6, frequeny=20, thld=2, stop=True):
    _tm = 1/frequeny
    ft_out = []
    curr_jnt = []
    print("total monitoring time: ", max_tm)
    # fz_gradient = []
    for i in np.arange(0, max_tm, _tm):
        ft = dc.get()
        ft = (ft-F0_reset)/sensitivity
        # ft[2] = 0 if ft[2] < 0 else ft[2]
        ft_out.append(ft)

            

        # print("@Pre: ", (ft-F0)/sensitivity)
        # print("@Pst: ", (ft-F0_reset)/sensitivity)
        # print("-----")

        # ft_out.append((ft-F0_reset)/sensitivity)
        # ft_out.append((ft-F0)/sensitivity)
        # ft_out.append(ft)
        
        if len(ft_out) < 10:
            ft_ = np.mean(ft_out, axis=0)
            fz = ft_[2]
        else:
            ft_ = np.mean(ft_out[-10:], axis=0)
            fz = ft_[2]
            # fz = ft_out_filter[:,2][-1]
            if fz > 1.5:
                print("[!] Stop! Fz=%.3f [N]" % fz)
                nxt.stopMotion()
                curr_jnt = nxt.getJointAngles()
                break
        time.sleep(_tm)

    ft_out = np.array(ft_out)

    # fig = plt.figure()
    # ax1 = fig.add_subplot(211)
    # ax1.axhline(y=1.2, color='gold', alpha=.7, linestyle='dashed')
    # ax1.plot(ft_out[:,2], alpha=0.3)
    # ax1.set_yticks(np.arange(-1,1,0.1))
    # ax2 = fig.add_subplot(212)
    # ax2.plot(fz_gradient)
    # print("If regrasp??: K2: ", np.sum(fz_gradient))
    # ax2.set_yticks(np.arange(-0.5,0.5,0.1))
    # ax2.axhline(y=-K1, color='gold', alpha=.7, linestyle='dashed')
    # ax2.axhline(y=K1, color='gold', alpha=.7, linestyle='dashed')
    # plt.show()
    return curr_jnt, ft_out
# def monitoring(max_tm=6, frequeny=20, thld=2, stop=True):
#     _tm = 1/frequeny
#     ft_out = []
#     curr_jnt = []
#     fz_gradient = []
#     for i in np.arange(0, max_tm, _tm):
#         ft = dc.get()
#         # print("@Pre: ", (ft-F0)/sensitivity)
#         # print("@Pst: ", (ft-F0_reset)/sensitivity)
#         # print("-----")

#         ft_out.append((ft-F0_reset)/sensitivity)
#         # ft_out.append((ft-F0)/sensitivity)
#         # ft_out.append(ft)
        
#         if len(ft_out) < 19:
#             ft_ = np.mean(ft_out, axis=0)
#             fz = ft_[2]
#         else:
#             ft_ = np.mean(ft_out[-10:], axis=0)
#             # _, ft_out_filter = fs.filter(np.arange(len(ft_out)), ft_out, w=15)
#             _, ft_out_filter = fs.smooth(np.arange(len(ft_out)), ft_out)
#             fz = ft_[2]
#             # fz = ft_out_filter[:,2][-1]
#             fz_prime = calc_gradient(ft_out_filter[:,2])[-1]
#             fz_gradient.append(fz_prime if fz_prime >= 0 else 0)
#                 # fz_ori.append(fz)
#             if fz_gradient != [] and stop and fz_gradient[-1] > K1:
#                 print("[!] Stop! Fz=%.3f [N]" % fz)
#                 nxt.stopMotion()
#                 curr_jnt = nxt.getJointAngles()
#                 break
#         time.sleep(_tm)
#     ft_out = np.array(ft_out)
#     fig = plt.figure()
#     ax1 = fig.add_subplot(211)
#     # ax1.plot(ft_out[:,2], alpha=0.3)
#     ax1.plot(ft_out_filter[:,2])
#     ax1.set_yticks(np.arange(-1,1,0.1))
#     ax2 = fig.add_subplot(212)
#     ax2.plot(fz_gradient)
#     print("If regrasp??: K2: ", np.sum(fz_gradient))
#     ax2.set_yticks(np.arange(-0.5,0.5,0.1))
#     ax2.axhline(y=-K1, color='gold', alpha=.7, linestyle='dashed')
#     ax2.axhline(y=K1, color='gold', alpha=.7, linestyle='dashed')
#     plt.show()
#     return curr_jnt, ft_out

pick(best_grasp_wrist)
clear_motion([])

jnt, ft, k4 = lift(best_grasp_wrist)
clear_motion(jnt)


if jnt != []:
    print("Untangling!!!")
    fling(j3=30, j4=40, vel=0.5)
    clear_motion([])
elif k4 < 1:
    print("regrasp!!!")
    regrasp()
    clear_motion([])
else:
    fling(j3=30, j4=40, vel=0.5)
    clear_motion([])

    # _, fout = monitoring(1, stop=False)
    # ft = np.mean(fout[:,-10:], axis=0)

# if close_to(ft[2], THLD_A, THLD_B):
#     jnt, ft = place()
#     clear_motion(jnt)
#     print("After placing: F=", ft)



    
#     print(f"After place | F={ft}")
    
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

    # _, fout = monitoring(2, stop=False)
    # ft = np.mean(fout[:,-10:], axis=0)
    # if ft[2] < 0.4 and close_to(ft[5], 0, delta=0.1):
    #     print("Regrasp!")
    # regrasp()

# not_finish = True

# while(not_finish):
#     print("Monitoring and calculate next action ... ")
#     _, fout = monitoring(2, stop=False)
#     fout_mean = dt.expectation(fout)
#     act = dt.infer([[fout_mean[i] for i in [2,5]]])
#     act_name = ['success', 'fling', 'regrasp']
#     print(act, act_name[int(act[0])])

#     app = DecisionWindow()
#     app.master.mainloop()
#     print("Supervised act: ", app.act)
#     print("Fine tuning decision tree! ")
#     dt.finetune([[fout_mean[i] for i in [2,5]]], [app.act])
#     with open(dt.add_path, 'a') as ap:
#         ap.write(np.append(fout_mean, app))
#     # np.savetxt(dt.add_path, np.append(fout_mean, app.act))

#     if app.act == 0: 
#         clear_motion([])
#         ret = place()
        
#         if ret is not None:
#             jnt, ft = ret
#             clear_motion(jnt)
#             if jnt != []:
#                 clear_motion(jnt)
#                 fling(j3=30, j4=40, vel=0.5)
#             else:
#                 break
#     elif app.act == 1:
#         clear_motion([])
#         fling(j3=30, j4=40, vel=0.5)

#     elif app.act == 2:
#         clear_motion([])
#         regrasp()

# put_ok()

    



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
