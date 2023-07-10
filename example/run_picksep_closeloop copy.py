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
# import scipy.interpolate as si

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

# read the predefined parameters
sensitivity = np.array(cfgdata["force_sensor"]["sensitivity"])

# arm = cfgdata["exp"]["lr_arm"]
arm = "right"
grasp_mode = cfgdata["exp"]["grasp_mode"]
log_mode = cfgdata["exp"]["log_mode"]
max_picking = cfgdata["exp"]["iteration"]
k_stop = cfgdata["exp"]["k_stop"]
k_fail = cfgdata["exp"]["k_fail"]
F0_reset = np.zeros(6)
N_max_t = 1

if log_mode:
    EXP_DIR = os.path.join("/home/hlab/Desktop", datetime.datetime.now().strftime('%Y%m%d%H%M%S'))
    os.mkdir(EXP_DIR)

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
        
        # best_grasp_tcp, eef_pose = transform_image_to_robot(best_grasp, point_array, cfgdata, arm=arm)
        # print("Pick | Grasp: (%d,%d,%.1f)" % (*best_grasp,)) 
        # print("Pick | TCP (%.3f,%.3f,%.3f), Wrist (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
        #             % (*best_grasp_tcp, *eef_pose)) 

        cv2.imwrite(draw_path, img_grasp)

        # tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        if log_mode:
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
        # nxt.playMotion(motion_seq)
        nxt.playMotion(motion_seq[:3])
        # time.sleep(0.5) 
        _ft0 = []
        set_zero_tm = 1
        for i in np.arange(0,set_zero_tm,0.05):
            if i > 0.1:
                _ft0.append(dc.get())
            time.sleep(0.05)
            
        global F0_reset
        
        F0_reset = np.mean(_ft0, axis=0).astype(int)
        # print("--- recalib F0:", F0_reset)
        nxt.playMotion(motion_seq[3:])


def lift(poses, tms):
    print("* * * * * * * * * *")
    print("*      Lift       *")
    print("* * * * * * * * * *")
    # pose_after = [0.500,-0.054,0.600,90,-90,-90]
    # success = plan_lift(arm, pose[:3], pose[3:], pose_after[:3], pose_after[3:])
    # success = plan_move(arm, [pose, pose_after], [3,6])
    # success = plan_move(arm, [pose_after], [4])
    # return [nothing, transport, swing, regrasp]=0,1,2,3
    
    success = plan_move(arm, poses, tms)
    
    motion_seq = get_motion()
    clear_motion()
    
    if not success: 
        raise SystemExit("Failed! ")
    elif PLAY: 
        # time.sleep(2)
        nxt.playMotion(motion_seq, wait=False)
        curr_jnt, out = monitoring(max_tm=np.sum(motion_seq[:,0]), f_thld=k_stop)
        if log_mode:
            tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            np.savetxt(os.path.join(EXP_DIR, 'lift_'+tstr+'.txt'), out)
            # np.savetxt("/home/hlab/Desktop/exp/"+tstr+"_lift.txt",fout) 
        popt = fs.fit(out[:,2], vis=False)
        # fs.plot(fout, filter=True)
        if np.abs(popt[0]) < 0.01:
            return Action.regrasp, curr_jnt
        elif curr_jnt != []: 
            return Action.swing, curr_jnt
        # elif np.mean(fout[20:,2]) < 0.1:
        #     return Action.putback, [] 
        else:
            return Action.transport, curr_jnt
    else: 
        return Action.swing, []
    #     return [], True

    
def regrasp(xyz_s, xyz_e):
    print("* * * * * * * * * *")
    print("*     Regrasp     *")
    print("* * * * * * * * * *")
        
    flip = False
    front_flip_pose = [*xyz_s,180,0,-90]
    back_flip_pose = [*xyz_s,0,0,-90]
    # front_flip_pose = [*xyz_s, 180,0,-90]
    # back_flip_pose = [0.480,-0.170,0.450,0,0,-90]
    pose_after = [*xyz_e,-102.4,-90,102.4]
    # pose_after = [0.500,-0.054,0.570,-102.4,-90,102.4]
    # success = plan_move(arm, front_flip_pose[:3], front_flip_pose[3:])
    
    if PLAY:
        success = plan_move(arm, [front_flip_pose], [2])
        motion_seq = get_motion()
        clear_motion()
        if not success:
            raise SystemExit("Failed! ")
        nxt.playMotion(motion_seq)
        _, out1 = monitoring(max_tm=0.75)
        # ft = np.mean(fout[:,-10:], axis=0)
        # ft_front = np.mean(fs.filter(fout)[1], axis=0)
        ft1 = np.mean(out1, axis=0)
        print("regrasp m1: ", ft1)
        # if ft_front[4] > -0.2: 
        #     print("May grasp nothing! ") 
            
        # if not close_to(ft[-1], 0, delta=0.03):
        # if ft_front[-1] > 0.03 or ft_front[-1] < -0.03: 
        #     flip = True
        #     print("FLIP!")
            
        # fs.plot(fout, filter=True)
        
        success = plan_move(arm, [back_flip_pose], [1])
        motion_seq = get_motion()
        clear_motion()

        if not success:
            raise SystemExit("Failed")
        nxt.playMotion(motion_seq)
        _, out2 = monitoring(max_tm=0.75)
        # ft_back = np.mean(fs.filter(fout)[1], axis=0)
        ft2 = np.mean(out2, axis=0)
        print("regrasp m2: ", ft2)

        # mx+my+mz instead of mz
        # if np.sum(np.abs(ft2[3:6])) < np.sum(np.abs(ft1[3:6])):
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

def put_ok():
    print("* * * * * * * * * *")
    print("*     Put ok      *")
    print("* * * * * * * * * *")
    # back_pose = [-0.050, -0.500, 0.450, 90, -90, -90]
    back_pose = [-0.100, -0.500, 0.450, 90, -90, -90]
    success = plan_put(arm, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
    clear_motion()
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq)

def swing(j, repeat=1):
    print("* * * * * * * * * *")
    print("*      Swing      *")
    print("* * * * * * * * * *")
    #TODO orientation and velocity
    # p = [0.480, -0.010, 0.480]
    p = [0.480, -0.010, 0.480]
    q = [90,-90,-90]

    # # print("Get real position: ", nxt.getJointPosition('RARM_JOINT5'))
    # print("Get sim  position: ", get_position('RARM_JOINT5'))
    # # raise SystemExit("Manually stopped! ")

    # p_ = get_position('RARM_JOINT5') 
    
    # if p_[2] > 0.48:
    #     print("Need to move to the initial pose for swingling! ")    
    #     p = p_
    # print(p)
    p_end = p.copy()
    # p_end[1] -= 0.30
    # p_end[2] += 0.2
    tm = 0.5
    j = [0,80,0]
    print("Move from", p, " to ", p_end)
    # success = plan_swing(arm, p, q, p_end, j, tm, repeat)
    success = plan_swing(arm, p, q, p_end, j, tm, 1, bilateral=False)
    motion_seq = get_motion()

    clear_motion()
    
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq[:1])
        # time.sleep(1)
        # delete the heavy entanglement swinging 
        # _, fout = monitoring(0.5, stop=False)
        # ft = np.mean(fout[:,-10:], axis=0)
        # print("Before swing ground truth", ft)
        # if ft[2] > 5: 
        #     print("Very Heavy TANGLE!")
        #     return
        nxt.playMotion(motion_seq[1:])
    # time.sleep(2)

def transport():
    print("* * * * * * * * * *")
    print("*    Transport    *")
    print("* * * * * * * * * *")
    # success = plan_transport(arm, [],[])
    # back_pose = [-0.050, -0.500, 0.550, 90, -90, -90]
    back_pose = [-0.200, -0.500, 0.600, 90, -90, -90]
    # success = plan_move(arm, back_pose[:3], back_pose[3:])
    success = plan_move(arm, [back_pose], [5])
    motion_seq = get_motion()
    clear_motion()

    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq, wait=False)

        curr_jnt, out = monitoring(max_tm=np.sum(motion_seq[:,0]), f_thld=k_stop)
        if log_mode:
            tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
            np.savetxt(os.path.join(EXP_DIR, 'transport_'+tstr+'.txt'), out)
        # np.savetxt("/home/hlab/Desktop/exp/"+tstr+"_transport.txt",out) 
        # fs.plot(fout, filter=True)
        if curr_jnt == []:
            return Action.transport, curr_jnt
        else:
            return Action.swing, curr_jnt
        # elif np.mean(fout[20:,2]) < 0.1:
        #     return Action.putback, [] 
        # else:
        #     return Action.standby, curr_jnt
    else:
        return Action.transport, []
        # return curr_jnt, np.mean(fout[-10:,:],axis=0)
    # else:
    #     return [], []

def spin(vel=1):
    success = plan_spin(arm, vel=vel)
    motion_seq = get_motion()
    clear_motion()

    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq, wait=False)


def monitoring(max_tm, frequeny=20, f_thld=None, filter=True):
    # default: max_tm = 6
    # print("............... Monitoring ...........")
    if not PLAY:
        return None, None, None
    _tm = 1/frequeny 
    out = []
    # fz_out = []
    jnt = []
    # fz_gradient = []
    for i in np.arange(0, max_tm, _tm):
        ft = dc.get()
        ft = (ft-F0_reset)/sensitivity
        # ft[2] = 0 if ft[2] < 0 else ft[2]
        out.append(ft)
        if i > 1 and f_thld is not None: # after 1 sec, start to filter 
            # detect emergency stop after 1 sec
        # if len(ft_out) < 10:
        #     # ft_ = np.mean(ft_out, axis=0)
        #     ft_ = np.mean(fs.filter(ft_out)[1], axis=0)
        #     fz = ft_[2]
        # else:
            # ft_ = np.mean(ft_out[-10:], axis=0)
            # ft_ = np.mean(fs.filter(ft_out[-10:])[1], axis=0)
            # ft_ = ft
            # fz = ft_[2]
            fz = ft[2]
            # fz = ft_out_filter[:,2][-1]
            if fz > f_thld:
                print("Stop! Fz=%.3f [N] > %.3f [N]" % (fz, f_thld))
                nxt.stopMotion()
                jnt = nxt.getJointAngles()
                break
            # fz_out.append(fz)
        time.sleep(_tm)

    # stopped_out = [] 
    # for i in np.arange(0, 1, _tm):
    #     ft = dc.get()
    #     ft = (ft-F0_reset)/sensitivity
    #     stopped_out.append(ft)
    #     time.sleep(_tm)

    if filter:
        _, out_ = fs.filter(out)
        return jnt, out_
    
    # print(ft_out)
    # ft_out = np.array(ft_out)
    # fs.plot(stopped_out, filter=True)
    return jnt, out

# ==================================== MAIN ==========================================
# j4_range = (30,35,40,45)
# j5_range = (40,45,50,55)
j4_range = (40,45,45,45)
j5_range = (50,55,55,55)
# j4_func = si.interp1d((0,1), np.array(j4_range))
# j5_func = si.interp1d((0,1), np.array(j5_range))


i = 0
l_fail, l_stop = [k_fail], [k_stop]
eef_pose_lifted = [0.500,-0.054,0.600,90,-90,-90]
# eef_pose_lifted = [0.500,-0.154,0.480,90,-90,-90]

while (i < max_picking):
    # initial parameters
    TRANSPORT_STOPPED = False
    LIFT_STOPPED = False
    curr_jnt = []
    N_transport = 0
    
    eef_pose = capture(method=grasp_mode)
    # eef_pose = [0.501, 0.052, 0.227, 90,-90,-90]

    pick(eef_pose)
    # regrasp(xyz_s=[0.480,-0.170,0.450], xyz_e=[0.500,-0.054,0.570])
    # regrasp(xyz_s=[0.400,-0.400,0.450], xyz_e=[0.500,-0.054,0.570])

    while (True):
        # os.system("bash /home/hlab/bpbot/script/start_ft.sh")
        # if N_transport == 0:
        #     eef_pose[2] += 0.05
        #     jnt, ft, is_linear = lift([eef_pose_lifted],[5])
        # else:
        #     jnt, ft, is_linear = lift([[0.500,-0.054,0.450,90,-90,-90],eef_pose_lifted],[3,3])
        if N_transport == 0: 
            action, jnt = lift([eef_pose_lifted],[5])
        else:
            action, jnt = lift([eef_pose_lifted],[3])
        

        # print("          ======== ", action.name, "=========")

        
        # ********** test action here ************* 
        # swing(j3=0,j4=30,j5=0,repeat=8)

        # spin(vel=0.5)
        # clear_motion()
        # regrasp()
        # clear_motion([])
        # print("Get real position: ", nxt.getJointPosition('Rarm_JOINT5'))
        # print("Get sim  position: ", get_position('Rarm_JOINT5'))

        # swing(j3=60, j4=55, vel=0.5)
        # # swing(j3=30, j4=40, vel=0.5)
        # clear_motion([])

        # regrasp()
        # swing(j3=j4_range[N_transport], j4=j5_range[N_transport], vel=0.5)
        # clear_motion([])
        # front_flip_pose = [0.480,-0.170,0.450,180,0,-90]
        # success = plan_move(arm, front_flip_pose[:3], front_flip_pose[3:])
        # motion_seq = get_motion()
        # if not success: raise SystemError("Failed")
        # elif PLAY:
        #     nxt.playMotion(motion_seq)
        # _, fout = monitoring(3, stop=False)
        # ft_front = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
        # print("Avg. FT: ", ft_front)

        # raise SystemExit("Exit here! ")
        # swing(j3=j4_range[N_transport], j4=j5_range[N_transport], vel=1)
        # clear_motion([])
        # ********** test action here ************* 

        # if jnt == [] and ft[2] < 0.1:

        # if action == Action.putback:
        #     print("grasp nothing! ")
        #     nxt.openHandToolLft()
        #     nxt.openHandToolRgt()
        #     nxt.setInitial(arm='all', tm=3)
        #     break

        # if (jnt==[] and is_linear and ft != [] and ft[2] < 0.3) or N_transport >= 2:
        if action == Action.regrasp or N_transport >= 2:
        # elif (jnt== [] and is_linear and ft[2] < 0.3) or (N_transport >= 2 and ft[2] < 1.5):
            # print("regrasp!!!")
            # temporal remove spin for longer ones
            # spin(vel=0.5)
            clear_motion()
            # regrasp for long cables
            regrasp(xyz_s=[0.420,-0.350,0.450], xyz_e=[0.500,-0.250,0.570])
            clear_motion()
            if PLAY:
                _, out = monitoring(1)
                ft = np.mean(out, axis=0)
                # ft = np.mean(fs.filter(fout)[1], axis=0)
                # ft = np.mean(fout[:,-10:], axis=0)
                # ft = np.mean(fout, axis=0)
                if ft[2] > k_fail: 
                    # os.system("bash /home/hlab/bpbot/script/start_ft.sh")
                    swing(j3=j4_range[N_transport], j4=j5_range[N_transport], vel=0.5)
                    clear_motion()

                # os.system("bash /home/hlab/bpbot/script/stop_ft.sh")
                # print("Additional, we record the force in ")
                # recorded_path = "/home/hlab/bpbot/data/force/raw_"+tstr+N_transport+".txt"
                # force_data = np.loadtxt("/home/hlab/bpbot/data/force/out.txt")
                # force_data2 = np.insert(force_data, 0, np.insert(F0_reset, 0, 0))
                # np.savetxt(recorded_path ,force_data2)

                else:
                    print("After grasping: some actions and transport")
                    # spin(vel=0.5)
                    # clear_motion()        
        
        # elif jnt != []:
        elif action == Action.swing:
            LIFT_STOPPED = True
            print("After grasping: swing")
            # os.system("bash /home/hlab/bpbot/script/start_ft.sh")
            # swing(j3=j4_range[N_transport], j4=j5_range[N_transport], j5=0, repeat=2)
            # swing(j3=45, j4=0, j5=0, repeat=2)
            swing([50,0,0],repeat=8)
            # swing(j3=30, j4=40, vel=0.5)
            clear_motion()

            # if PLAY:
            #     _, fout = monitoring(1, stop=False)
            #     # ft = np.mean(fout[:,-10:], axis=0)
            #     # ft = np.mean(fs.filter(np.arange(len(fout)), fout, method="median", param=11)[1], axis=0)
            #     ft = np.mean(fs.filter(fout)[1], axis=0)
            #     print("After swing value ", ft)
            #     if ft[2] < 0.2:
            #         print("grasp nothing! ")
            #         nxt.openHandToolLft()
            #         nxt.openHandToolRgt()
            #         nxt.setInitial(arm='all', tm=3)
            #         break

        # else:
        #     spin(vel=0.5)
        #     clear_motion()        
        # check again 

        
        action, jnt = transport()
        clear_motion(jnt)
        
        
        # if jnt == [] and ft[2] <= k_fail + 0.1: 
        if action == Action.transport:
            if PLAY:
                # _, fout = monitoring(1, stop=False)
                _, out = monitoring(1, filter=True)
                # ft = np.mean(fout[:,-10:], axis=0)
                # ft = np.mean(fs.filter(np.arange(len(fout)), fout, method="median", param=11)[1], axis=0)
                # ft = np.mean(fs.filter(fout)[1], axis=0)
                ft = np.mean(out, axis=0)

                print("Adjust k_fail", k_fail, 'N -> ', end='')
                l_fail.append(ft[2])
                print("l_fail: ", l_fail)
                # k_fail = np.mean(l_fail)
                print(k_fail, "[N]")
            # np.savetxt("/home/hlab/Desktop/ffail.txt", l_fail)
            put_ok()
            break
        # elif ft[2] < 0.1:
        # elif action == Action.putback:
        elif action == Action.swing:
            if PLAY:
                swing([50,0,0],repeat=8)
        else:
            regrasp(xyz_s=[0.420,-0.350,0.450], xyz_e=[0.500,-0.250,0.570])
            clear_motion()
            # print("Grasp nothing")
            # put_back()
        # else:
        #     print("Transport failed, try again")
        #     N_transport += 1

        # if  jnt != [] and LIFT_STOPPED != False:
        #     print("Here is timing to decrease k_stop: ", k_stop)
        #     # k_stop -= 0.1
        #     print(" -> ", k_stop)
        
        print("# Swing : ", N_transport)
        # if N_transport >= N_max_t:
        #     put_back()
        #     break
        
        # if jnt == [] and ft[2] > k_fail: 
        #     print("Detected grasping two objects. put back and startover!!!!!")
        #     put_back()
        #     break

    print("Finish one picking attempt! ")
    i+=1

# ==================================== MAIN OVER =====================================