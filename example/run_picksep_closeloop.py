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
import scipy.interpolate as si

from enum import Enum
class Action(Enum):
    standby = -1
    putback = 0
    transport = 1
    swing = 2
    regrasp = 3
    putok = 4

nxt = NxtRobot(host='[::]:15005')
dc = DynPickClient()
fs = DynPickControl()

root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
img_path = os.path.join(root_dir, "data/depth/depth.png")
crop_path = os.path.join(root_dir, "data/depth/cropped.png")
draw_path = os.path.join(root_dir, "data/depth/result.png")

config_path = os.path.join(root_dir, "config/config_picksep_cable.yaml")
cfg = BinConfig(config_path=config_path)
cfgdata = cfg.data

sensitivity = np.array(cfgdata["force_sensor"]["sensitivity"])

arm = "right"
grasp_mode = cfgdata["exp"]["grasp_mode"]
log_mode = cfgdata["exp"]["log_mode"]
max_picking = cfgdata["exp"]["iteration"]
k_stop = cfgdata["exp"]["k_stop"]
k_fail = cfgdata["exp"]["k_fail"]
F0_reset = np.zeros(6)
N_max_t = 1


PLAY = cfgdata["exp"]["play_mode"]

def range_in(x, lower, upper):
    return True if x < upper and x > lower else False

def close_to(x, v, delta=1.01):
    return True if x > v-delta and x < v+delta else False

def calc_gradient(l):
    g = []
    size = len(l)
    for i in range(size-1):
        # g.append(l[size-1-i]-l[size-2-i])
        g.append(l[i+1]-l[i])
    return np.array(g)

def capture(method="fge"): 
    print("* * * * * * * * * *")
    print("*     Capture     *")
    print("* * * * * * * * * *")
    
    point_array = capture_pc()
    if point_array is not None:
        img, img_blur = pc2depth(point_array, cfgdata)
        cv2.imwrite(img_path, img_blur)
        
        crop = crop_roi(img_blur, cfgdata)

        cv2.imwrite(crop_path, crop)

        if method == "fge":
            grasps = detect_grasp(n_grasp=5, 
                                img_path=crop_path, 
                                g_params=cfgdata['graspability'],
                                h_params=cfgdata["hand"][arm])
            best_grasp_idx = 0
            img_grasp = draw_grasp(grasps, crop_path, cfgdata["hand"][arm], top_only=True, top_idx=best_grasp_idx)
        elif method == "pn":
            grasps = picknet(img_path=crop_path, hand_config=cfgdata["hand"][arm])
            best_grasp_idx = 0
        
        elif method == "asp":
            grasps = detect_grasp(n_grasp=5, 
                                img_path=crop_path, 
                                g_params=cfgdata['graspability'],
                                h_params=cfgdata["hand"][arm])
            grasp_pixels = np.array(grasps)[:, 0:2]
            _, best_grasp_idx = predict_action_grasp(grasp_pixels, crop_path)
            drawn_g = [grasps[i] for i in [best_grasp_idx, 0]]
            img_grasp = draw_grasp(drawn_g, crop_path, cfgdata["hand"][arm], top_only=False)

        if grasps is None:
            print(f"No grasp detected! ")
            raise SystemExit("Failed! ")
        
        best_grasp = grasps[best_grasp_idx]
        
        obj_pose, eef_pose = transform_image_to_robot(best_grasp, point_array, cfgdata, arm=arm)
        print("Grasp | Object 3D location: (%.3f,%.3f,%.3f)" % (*obj_pose,))
        print("Grasp | Robot EEF pose (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" % (*eef_pose,)) 
        
        cv2.imwrite(draw_path, img_grasp)

        if log_mode:
            np.savetxt(os.path.join(EXP_DIR, "log.txt"), obj_pose)
            cv2.imwrite(os.path.join(EXP_DIR, "img.png"), crop)
            cv2.imwrite(os.path.join(EXP_DIR, "ret.png"), img_grasp)
            
        return eef_pose
    else:
        print("No camera available! Return a fix eef pose")
        return [0.48,0.20,0.20,0,-90,-90]

def pick(pose):
    global ft_reset
    print("* * * * * * * * * *")
    print("*      Pick       *")
    print("* * * * * * * * * *")
    
    success = plan_pick(arm, pose[:3], pose[3:], init=True)
    motion_seq = get_motion()
    clear_motion()
    if not success: 
        raise SystemExit("Failed! ")
    if PLAY: 
        nxt.playMotion(motion_seq[:3])
        _ft0 = []
        set_zero_tm = 1
        for i in np.arange(0,set_zero_tm,0.05):
            if i > 0.1:
                _ft0.append(dc.get())
            time.sleep(0.05)
            
        global F0_reset
        
        F0_reset = np.mean(_ft0, axis=0).astype(int)
        nxt.playMotion(motion_seq[3:])


def lift(poses, tms, stop=True):
    print("* * * * * * * * * *")
    print("*      Lift       *")
    print("* * * * * * * * * *")
    
    success = plan_move(arm, poses, tms)
    motion_seq = get_motion()
    
    if not success: 
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq, wait=False)
        jnt, out = monitoring(max_tm=np.sum(motion_seq[:,0]), f_thld=k_stop)
        clear_motion(jnt)
        
        if log_mode:
            tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            np.savetxt(os.path.join(EXP_DIR, 'lift_'+tstr+'.txt'), out)

        if jnt != []:
            return Action.swing
        else:
            if stop: 
                popt = fs.fit(out[:,2], vis=True)
                print("Fitted results:", *popt, "threshold: ", k_fail/0.4)
                # if ft_static[2] < k_fail and np.abs(popt[0]) > 0.01:
                if popt[0] < 0.1: 
                    print("grasp nothing! ")
                    nxt.openHandToolLft()
                    nxt.openHandToolRgt()
                    nxt.setInitial(arm='all', tm=3)
                    clear_motion()
                    return

                elif popt[0] < 0.5:
                    return Action.regrasp
                # elif ft_static[2] > k_fail:
                #     return Action.swing
                else:
                    return Action.transport
            else:
                _, out_static = monitoring(max_tm=1)
                ft_static = np.mean(out_static, axis=0)
                print("Finish lifting, current force/torque", ft_static)
                if ft_static[2] > k_fail:
                    return Action.swing
                else:
                    return Action.transport

            
            # if ft_static[2] < k_fail and np.abs(popt[0]) > 0.01:
            #     return Action.transport
            # else:
            #     return Action.regrasp

        # popt = fs.fit(out[:,2], vis=False)
        # # fs.plot(out, filter=True)
        # if np.abs(popt[0]) < 0.01:
        #     return Action.regrasp
        # elif jnt != []: 
        #     return Action.swing
        # # elif np.mean(out[20:,2]) < 0.1:
        # #     return Action.putback, [] 
        # else:
        #     return Action.transport
    else: 
        clear_motion()
        # return Action.regrasp
        return Action.swing

    
def regrasp(xyz_s, xyz_e):
    print("* * * * * * * * * *")
    print("*     Regrasp     *")
    print("* * * * * * * * * *")
        
    front_flip_pose = [*xyz_s,180,0,-90]
    back_flip_pose = [*xyz_s,0,0,-90]
    pose_after = [*xyz_e,-102.4,-90,102.4]
    
    if PLAY:
        success = plan_move(arm, [front_flip_pose], [2])
        motion_seq = get_motion()
        clear_motion()
        if not success:
            raise SystemExit("Failed! ")
        nxt.playMotion(motion_seq)
        _, out1 = monitoring(max_tm=0.75)
        ft1 = np.mean(out1, axis=0)
        print("regrasp m1: ", ft1)
        # fs.plot(fout, filter=True)
        
        success = plan_move(arm, [back_flip_pose], [1])
        motion_seq = get_motion()
        clear_motion()

        if not success:
            raise SystemExit("Failed")
        nxt.playMotion(motion_seq)
        _, out2 = monitoring(max_tm=0.75)
        ft2 = np.mean(out2, axis=0)
        print("regrasp m2: ", ft2)

        if np.abs(ft2[-1]) < np.abs(ft1[-1]):
        # if np.abs(ft_back[5]) < np.abs(ft_front[5]):
            print("Proceed!")
            success = plan_regrasp(back_flip_pose[:3], back_flip_pose[3:], pose_after[:3], pose_after[3:])
        else:
            print("Flip back! ")
            success = plan_regrasp(front_flip_pose[:3], front_flip_pose[3:], pose_after[:3], pose_after[3:])

        motion_seq = get_motion()
        clear_motion()
        if not success:
            raise SystemExit("Failed! ")

        nxt.playMotion(motion_seq)

    if not PLAY:
        success = plan_regrasp(back_flip_pose[:3], back_flip_pose[3:], pose_after[:3], pose_after[3:])
        motion_seq = get_motion()
        clear_motion()

def put_back():
    print("* * * * * * * * * *")
    print("*     Put back    *")
    print("* * * * * * * * * *")
    back_pose = [0.480, -0.010, 0.480, 90, -90, -90]
    success = plan_put(arm, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
    clear_motion()
    if not success:
        raise SystemExit("[!] Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq)

def put_ok(pose):
    print("* * * * * * * * * *")
    print("*     Put ok      *")
    print("* * * * * * * * * *")
    # back_pose = [-0.050, -0.500, 0.450, 90, -90, -90]
    pose[2] = 0.45
    success = plan_put(arm, pose[:3], pose[3:])
    motion_seq = get_motion()
    clear_motion()
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq)

def swing(pose, j, repeat=1):
    print("* * * * * * * * * *")
    print("*      Swing      *")
    print("* * * * * * * * * *")
    p = pose[:3]
    q = pose[3:]

    p_end = p.copy()
    # p_end[1] -= 0.30
    # p_end[2] += 0.2
    tm = 0.5
    j = [40,70,0]
    # j = [0,60,40]
    repeat = 1
    
    success = plan_swing(arm, p, q, p_end, j, tm, repeat, bilateral=False)
    motion_seq = get_motion()

    clear_motion()
    
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq)

def transport(pose):
    print("* * * * * * * * * *")
    print("*    Transport    *")
    print("* * * * * * * * * *")
    
    success = plan_move(arm, [pose], [5])
    motion_seq = get_motion()

    if not success:
        raise SystemExit("Failed! ")
    if PLAY: 
        _, out = monitoring(1, filter=True)
        ft = np.mean(out, axis=0)
        # print("Force before transporting: ", ft)
        # if ft[2] < 0.1:
        #     nxt.openHandToolLft()
        #     nxt.openHandToolRgt()
        #     nxt.setInitial(arm='all', tm=3)
        #     return Action.standby

        nxt.playMotion(motion_seq, wait=False)

        jnt, out = monitoring(max_tm=np.sum(motion_seq[:,0]), f_thld=k_stop)
        # fs.plot(out, filter=True)
        clear_motion(jnt)

        if log_mode:
            tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            np.savetxt(os.path.join(EXP_DIR, 'transport_'+tstr+'.txt'), out)

        if jnt == []:
            return Action.transport
        else:
            _, out = monitoring(1, filter=True)
            ft = np.mean(out, axis=0)
            if ft[2] > 1:
                return Action.swing
    else:
        clear_motion()
        return Action.transport

def spin(vel=1):
    success = plan_spin(arm, vel=vel)
    motion_seq = get_motion()
    clear_motion()

    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq, wait=False)


def monitoring(max_tm, frequeny=20, f_thld=None, filter=True):
    if not PLAY:
        return None, None
    _tm = 1/frequeny 
    out = []
    jnt = []
    for i in np.arange(0, max_tm, _tm):
        ft = dc.get()
        ft = (ft-F0_reset)/sensitivity
        out.append(ft)
        if i > 1 and f_thld is not None: # after 1 sec, start to filter 
            fz = ft[2]
            if fz > f_thld:
                print("Stop! Fz=%.3f [N] > %.3f [N]" % (fz, f_thld))
                nxt.stopMotion()
                jnt = nxt.getJointAngles()
                break
        time.sleep(_tm)

    if filter:
        _, out_ = fs.filter(out)
        return jnt, out_
    
    return jnt, out

# j4_range = (40,45,45,45)
# j5_range = (50,55,55,55)
j3_range = [45,60]
j4_range = [60,80]
j5_range = [60,80]
j3 = si.interp1d((0,1), np.array(j3_range))
j4 = si.interp1d((0,1), np.array(j4_range))
j5 = si.interp1d((0,1), np.array(j5_range))

i = 0
l_fail, l_stop = [k_fail], [k_stop]
eef_pose_lifted = [0.500,-0.054,0.600,90,-90,-90]
eef_pose_transport = [-0.200, -0.500, 0.600, 90, -90, -90]
eef_pose_swing = [0.460, -0.010, 0.550, 90,-90,-90] # vertical
# eef_pose_swing = [0.460, -0.010, 0.400, 180,-60,-180] # tilt
eef_xyz_regrasp_start = [0.480, -0.170, 0.450]
eef_xyz_regrasp_end = [0.500, -0.250, 0.570]
# eef_pose_lifted = [0.500,-0.154,0.480,90,-90,-90]

while (i < max_picking):
    if log_mode:
        EXP_DIR = os.path.join("/home/hlab/Desktop", "exp",datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
        os.mkdir(EXP_DIR)
    TRANSPORT_STOPPED = False
    LIFT_STOPPED = False
    curr_jnt = []
    N_transport = 0
    N_swing = 0
    N_regrasp = 0
    
    eef_pose = capture(method=grasp_mode)
    pick(eef_pose)

    while (True):
        if N_transport == 0: 
            action = lift([eef_pose_lifted],[5])
        else:
            action = lift([eef_pose_lifted],[3], stop=False)
        
        # spin(vel=0.5)
        # regrasp(eef_xyz_regrasp_start, eef_xyz_regrasp_end)
        # raise SystemExit("Demo! ")
        if action == Action.regrasp or N_transport >= 2 or N_swing >= 2:
            # regrasp(xyz_s=[0.420,-0.350,0.450], xyz_e=[0.500,-0.250,0.570])
            regrasp(eef_xyz_regrasp_start, eef_xyz_regrasp_end)
            N_regrasp += 1
            spin(vel=0.5)
            # swing(eef_pose_swing,[50,0,0],repeat=8)
            N_swing += 1
            
            # if PLAY:
            #     _, out = monitoring(1)

            #     ft = np.mean(out, axis=0)
            #     print("Force/torque output: ", ft)
            #     if ft[2] > k_fail: 
            #         swing(eef_pose_swing,j=[0,j4_range[N_transport], j5_range[N_transport]])
            #         N_swing += 1

            #     else:
            #         print("After grasping: some actions and transport")
     
        elif action == Action.swing:
            LIFT_STOPPED = True
            # spin(vel=0.5)
            _, out = monitoring(1, filter=True)
            ft = np.mean(out, axis=0)
            torque = np.sum(ft[3:])
            sw_idx = 1 if N_swing*0.25 > 1 else N_swing*0.25
            if torque > 0:
                swing_param = [0, j4(sw_idx), j5(sw_idx)]
            else:
                swing_param = [0, j4(sw_idx), -j5(sw_idx)]
            swing(eef_pose_swing,swing_param,repeat=8)
            swing(eef_pose_swing,[50,0,0],repeat=8)
            N_swing += 1
        
        spin(vel=0.5)
        action = transport(eef_pose_transport)
        N_transport += 1
        
        if action == Action.transport:
            # if PLAY:
            #     _, out = monitoring(1, filter=True)
            #     ft = np.mean(out, axis=0)
            #     print("TODO: parameter tuning! ")
            #     if ft[2] < 0.3:
            #         nxt.openHandToolLft()
            #         nxt.openHandToolRgt()
            #         nxt.setInitial(arm='all', tm=3)
            #         break

                # print("Adjust k_fail", k_fail, 'N -> ', end='')
                # l_fail.append(ft[2])
                # print("l_fail: ", l_fail)
                # print(k_fail, "[N]")

            put_ok(eef_pose_transport)
            if log_mode:
                with open(os.path.join(EXP_DIR, "log.txt"), 'a') as fp:
                    print(f"Swing:{N_swing}\nRegrasp:{N_regrasp}\n", file=fp)
            break
        elif action == Action.standby:
            # if  LIFT_STOPPED != False:
            #     print("Here is timing to decrease k_stop: ", k_stop)
            #     k_stop -= 0.1
            #     print(" -> ", k_stop)
            break
        # elif action == Action.swing:
        
        # elif action == Action.standby:
        #     if PLAY:
        #         _, out = monitoring(1, filter=True)
        #         ft = np.mean(out, axis=0)

        # elif action == Action.swing:
        #     swing(eef_pose_swing, [50,20,0],repeat=8)
        #     N_swing += 1
        # else:
        #     regrasp(xyz_s=[0.420,-0.350,0.450], xyz_e=[0.500,-0.250,0.570])

        # print("# Swing : ", N_transport)

    print("Finish one picking attempt! ")
    i+=1
