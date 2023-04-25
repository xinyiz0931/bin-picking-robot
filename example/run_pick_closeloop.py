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

# global arguments
PLAY = True

# METHOD = "asp"
METHOD = "fge"
MAX_I = 1

ARM = "right"
sensitivity = np.array([32.800,32.815,32.835,1653.801,1634.816,1636.136])
F0_reset = np.zeros(6)

tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')

nxt = NxtRobot(host='[::]:15005')
dc = DynPickClient()
fs = DynPickControl()
# for tuning
k_one = 0.5 
k_stop = 2
# k_stop = 1
k_fail = 0.7


from multiprocessing import Process
lis = []
def foo(i):
    lis.append(i)
    print("This is Process ", i," and lis is ", lis, " and lis.address is  ", id(lis))

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

def capture(method="fge"): 
    # initialize
    root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
    print(f"Execute script at {root_dir} ")

    img_path = os.path.join(root_dir, "data/depth/depth.png")
    crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
    draw_path = os.path.join(root_dir, "data/depth/result.png")

    # ---------------------- get config info -------------------------
    cfg = BinConfig()
    cfgdata = cfg.data
    
    point_array = capture_pc()
    if point_array is not None:
        print("* * * * * * * * * *")
        print("*     Capture     *")
        print("* * * * * * * * * *")
        img, img_blur = pc2depth(point_array, cfgdata)
        cv2.imwrite(img_path, img_blur)
        
        crop = crop_roi(img_blur, cfgdata)

        cv2.imwrite(crop_path, crop)

        if method == "fge":
        
            grasps = detect_grasp(n_grasp=5, 
                                img_path=crop_path, 
                                g_params=cfgdata['graspability'],
                                h_params=cfgdata["hand"][ARM])
            best_grasp_idx = 0
            img_grasp = draw_grasp(grasps, crop_path, cfgdata["hand"][ARM], top_only=True, top_idx=best_grasp_idx)
        elif method == "pn":
            grasps = picknet(img_path=crop_path, hand_config=cfgdata["hand"][ARM])
            print(grasps)
            best_grasp_idx = 0
        
        elif method == "asp":
            grasps = detect_grasp(n_grasp=5, 
                                img_path=crop_path, 
                                g_params=cfgdata['graspability'],
                                h_params=cfgdata["hand"][ARM])
            grasp_pixels = np.array(grasps)[:, 0:2]
            _, best_grasp_idx = predict_action_grasp(grasp_pixels, crop_path)
            drawn_g = [grasps[i] for i in [best_grasp_idx, 0]]
            img_grasp = draw_grasp(drawn_g, crop_path, cfgdata["hand"][ARM], top_only=False)
        if grasps is None:
            print(f"No grasp detected! ")
            raise SystemExit("Failed! ")
        
        best_grasp = grasps[best_grasp_idx]
        
        best_grasp_tcp, best_grasp_wrist = transform_image_to_robot(best_grasp, point_array, cfgdata, hand=ARM)
        print("Pick | Grasp: (%d,%d,%.1f)" % (*best_grasp,)) 
        print("Pick | TCP (%.3f,%.3f,%.3f), Wrist (%.3f,%.3f,%.3f,%.1f,%.1f,%.1f)" 
                    % (*best_grasp_tcp, *best_grasp_wrist)) 

        cv2.imwrite(draw_path, img_grasp)

        # tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        # cv2.imwrite("/home/hlab/Desktop/exp/"+tstr+"_img.png", crop)
        # cv2.imwrite("/home/hlab/Desktop/exp/"+tstr+"_ret.png", img_grasp)
    return best_grasp_wrist


def pick(pose):
    global ft_reset
    print("* * * * * * * * * *")
    print("*      Pick       *")
    print("* * * * * * * * * *")
    
    success = plan_pick(ARM, pose[:3], pose[3:], init=True)
    motion_seq = get_motion()
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
        print("--- recalib F0:", F0_reset)

        nxt.playMotion(motion_seq[3:])


def lift(poses, tms):
    print("* * * * * * * * * *")
    print("*      Lift       *")
    print("* * * * * * * * * *")
    # pose_after = [0.500,-0.054,0.600,90,-90,-90]
    # success = plan_lift(ARM, pose[:3], pose[3:], pose_after[:3], pose_after[3:])
    # success = plan_move(ARM, [pose, pose_after], [3,6])
    # success = plan_move(ARM, [pose_after], [4])
    
    success = plan_move(ARM, poses, tms)
    
    motion_seq = get_motion()
    
    if not success: 
        raise SystemExit("Failed! ")
    elif PLAY: 
        # time.sleep(2)
        nxt.playMotion(motion_seq, wait=False)
        curr_jnt, fout = monitoring(max_tm=np.sum(motion_seq[:,0]), f_thld=k_stop)
        tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        np.savetxt("/home/hlab/Desktop/exp/"+tstr+"_lift.txt",fout) 
        is_linear = fs.fit(fout[:,2], vis=False)

        # fs.plot(fout, filter=True)
        return curr_jnt, np.mean(fout[-20:,:],axis=0), is_linear
    else:
        return [],[], True

    
def regrasp():
    print("* * * * * * * * * *")
    print("*     Regrasp     *")
    print("* * * * * * * * * *")
    flip = False
    front_flip_pose = [0.480,-0.170,0.450,180,0,-90]
    back_flip_pose = [0.480,-0.170,0.450,0,0,-90]
    pose_after = [0.500,-0.054,0.570,-102.4,-90,102.4]
    # success = plan_move(ARM, front_flip_pose[:3], front_flip_pose[3:])
    success = plan_move(ARM, [front_flip_pose], [2])
    motion_seq = get_motion()
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY:
        nxt.playMotion(motion_seq)
        _, fout = monitoring(max_tm=0.5, stop=False)
        # ft = np.mean(fout[:,-10:], axis=0)
        ft_front = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
        print("regrasp m1: ", ft_front)
        if ft_front[4] > -0.2: 
            print("May grasp nothing! ") 

            
        # if not close_to(ft[-1], 0, delta=0.03):
        # if ft_front[-1] > 0.03 or ft_front[-1] < -0.03: 
        #     flip = True
        #     print("FLIP!")
            
        # fs.plot(fout, filter=True)
    
        clear_motion([])
        
        success = plan_move(ARM, [back_flip_pose], [1])
        motion_seq = get_motion()
        if not success:
            raise SystemExit("Failed")
        elif PLAY: 
            nxt.playMotion(motion_seq)
            _, fout = monitoring(max_tm=0.5, stop=False)
            ft_back = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
            print("regrasp m2: ", ft_back)

            clear_motion([])
            # mx+my+mz instead of mz
            if np.sum(np.abs(ft_back[3:6])) < np.sum(np.abs(ft_front[3:6])):
            # if np.abs(ft_back[5]) < np.abs(ft_front[5]):
                print("Proceed!")
                success = plan_regrasp(back_flip_pose[:3], back_flip_pose[3:], pose_after[:3], pose_after[3:])
            else:
                print("Flip back! ")
                success = plan_regrasp(front_flip_pose[:3], front_flip_pose[3:], pose_after[:3], pose_after[3:])

            motion_seq = get_motion()
            if not success:
                raise SystemExit("Failed! ")

            elif PLAY:
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
    print("* * * * * * * * * *")
    print("*     Put back    *")
    print("* * * * * * * * * *")
    back_pose = [0.480, -0.010, 0.480, 90, -90, -90]
    success = plan_put(ARM, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
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
    success = plan_put(ARM, back_pose[:3], back_pose[3:])
    motion_seq = get_motion()
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq)

def fling(j3, j4, vel=1):
    print("* * * * * * * * * *")
    print("*      Fling      *")
    print("* * * * * * * * * *")
    #TODO orientation and velocity
    p = [0.480, -0.010, 0.480]
    q = [90,-90,-90]

    p_ = get_position('RARM_JOINT5') 

    print("Get sim  position: ", get_position('RARM_JOINT5'))
    
    if p_[2] > 0.48:
        print("Need to move to the initial pose for flingling! ")    
        p = p_

    success = plan_fling(ARM, p, q, j3, j4, vel)
    motion_seq = get_motion()

    
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq[:1])
        # time.sleep(1)
        # delete the heavy entanglement flinging 
        # _, fout = monitoring(0.5, stop=False)
        # ft = np.mean(fout[:,-10:], axis=0)
        # print("Before fling ground truth", ft)
        # if ft[2] > 5: 
        #     print("Very Heavy TANGLE!")
        #     return
        nxt.playMotion(motion_seq[1:])
    # time.sleep(2)

def transport():
    print("* * * * * * * * * *")
    print("*    Transport    *")
    print("* * * * * * * * * *")
    # success = plan_transport(ARM, [],[])
    # back_pose = [-0.050, -0.500, 0.550, 90, -90, -90]
    back_pose = [-0.100, -0.500, 0.600, 90, -90, -90]
    # success = plan_move(ARM, back_pose[:3], back_pose[3:])
    success = plan_move(ARM, [back_pose], [5])
    motion_seq = get_motion()
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq, wait=False)

        curr_jnt, fout = monitoring(max_tm=np.sum(motion_seq[:,0]), f_thld=k_stop)
        tstr = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
        np.savetxt("/home/hlab/Desktop/exp/"+tstr+"_transport.txt",fout) 
        # fs.plot(fout, filter=True)
        return curr_jnt, np.mean(fout[-10:,:],axis=0)
    else:
        return [], []

def spin(vel=1):
    success = plan_spin(ARM, vel=vel)
    motion_seq = get_motion()
    if not success:
        raise SystemExit("Failed! ")
    elif PLAY: 
        nxt.playMotion(motion_seq, wait=False)


def monitoring(max_tm=6, frequeny=20, f_thld=2, stop=True):
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
        
        # ft_out_filter = fs.filter(np.arange(len(ft_out)), ft_out, method="median", param=7)

        if len(ft_out) < 10:
            # ft_ = np.mean(ft_out, axis=0)
            ft_ = np.mean(fs.filter(np.arange(len(ft_out)), ft_out, method="median", param=11)[1], axis=0)
            fz = ft_[2]
        else:
            # ft_ = np.mean(ft_out[-10:], axis=0)
            ft_ = np.mean(fs.filter(np.arange(10), ft_out[-10:], method="median", param=11)[1], axis=0)
            fz = ft_[2]
            # fz = ft_out_filter[:,2][-1]
            if fz > f_thld:
                print("Stop! Fz=%.3f [N] > %.3f [N]" % (fz, f_thld))
                nxt.stopMotion()
                curr_jnt = nxt.getJointAngles()
                break
        time.sleep(_tm)
    time.sleep(0.5)

    ft_out = np.array(ft_out)
    return curr_jnt, ft_out

# ==================================== MAIN ==========================================
# j4_range = (30,35,40,45)
# j5_range = (40,45,50,55)
j4_range = (40,45,45,45)
j5_range = (50,55,55,55)
# j4_func = si.interp1d((0,1), np.array(j4_range))
# j5_func = si.interp1d((0,1), np.array(j5_range))


i = 0
l_fail, l_stop = [k_fail], [k_stop]
pose_after_lift = [0.500,-0.054,0.600,90,-90,-90]

while (i < MAX_I):
    # initial parameters
    TRANSPORT_STOPPED = False
    LIFT_STOPPED = False
    curr_jnt = []
    N_transport = 0
    
    best_grasp_wrist = capture(method=METHOD)
    # best_grasp_wrist = [0.501, 0.052, 0.227, 90,-90,-90]

    pick(best_grasp_wrist)
    clear_motion([])

    while (True):
        os.system("bash /home/hlab/bpbot/script/start_ft.sh")
        if N_transport == 0:
            best_grasp_wrist[2] += 0.05
            jnt, ft, is_linear = lift([pose_after_lift], [5])
        else:
            # jnt, ft, is_linear = lift([pose_after_lift], [3])
            # jnt, ft
            jnt, ft, is_linear = lift([[0.500,-0.054,0.450,90,-90,-90],pose_after_lift], [3,3])
        clear_motion(jnt)
        print("F/T output: ", ft)
        # print("Get real position: ", nxt.getJointPosition('RARM_JOINT5'))
        # print("Get sim  position: ", get_position('RARM_JOINT5'))

        # ********** test action here ************* 
        fling(j3=60, j4=55, vel=0.5)
        # fling(j3=30, j4=40, vel=0.5)
        clear_motion([])

        os.system("bash /home/hlab/bpbot/script/stop_ft.sh")
        print("Additional, we record the force in ")
        recorded_path = "/home/hlab/bpbot/data/force/raw_"+str(tstr)+"_"+str(N_transport)+".txt"
        force_data = np.loadtxt("/home/hlab/bpbot/data/force/out.txt")
        force_data2 = np.insert(force_data, 0, np.insert(F0_reset, 0, 0), axis=0)
        np.savetxt(recorded_path ,force_data2)

        raise SystemExit("out!")
        # regrasp()
        # fling(j3=j4_range[N_transport], j4=j5_range[N_transport], vel=0.5)
        # clear_motion([])
        # front_flip_pose = [0.480,-0.170,0.450,180,0,-90]
        # success = plan_move(ARM, front_flip_pose[:3], front_flip_pose[3:])
        # motion_seq = get_motion()
        # if not success: raise SystemError("Failed")
        # elif PLAY:
        #     nxt.playMotion(motion_seq)
        # _, fout = monitoring(3, stop=False)
        # ft_front = np.mean(fs.filter(np.arange(fout.shape[0]), fout)[1], axis=0)
        # print("Avg. FT: ", ft_front)

        # raise SystemExit("Exit here! ")
        # fling(j3=j4_range[N_transport], j4=j5_range[N_transport], vel=1)
        # clear_motion([])
        # ********** test action here ************* 

        if jnt == [] and ft[2] < 0.1:
            print("grasp nothing! ")
            nxt.openHandToolLft()
            nxt.openHandToolRgt()
            nxt.setInitial(arm='all', tm=3)
            break
        # if (jnt==[] and is_linear and ft != [] and ft[2] < 0.3) or N_transport >= 2:
        elif (jnt== [] and is_linear and ft[2] < 0.3) or (N_transport >= 2 and ft[2] < 1.5):
            print("regrasp!!!")
            spin(vel=0.5)
            clear_motion([])
            regrasp()
            clear_motion([])
            _, fout = monitoring(1, stop=False)
            ft = np.mean(fs.filter(np.arange(len(fout)), fout, method="median", param=11)[1], axis=0)
            # ft = np.mean(fout[:,-10:], axis=0)
            # ft = np.mean(fout, axis=0)
            if ft[2] > k_fail: 
                print("After grasping: fling")
                os.system("bash /home/hlab/bpbot/script/start_ft.sh")
                fling(j3=j4_range[N_transport], j4=j5_range[N_transport], vel=0.5)
                clear_motion([])

                os.system("bash /home/hlab/bpbot/script/stop_ft.sh")
                print("Additional, we record the force in ")
                recorded_path = "/home/hlab/bpbot/data/force/raw_"+tstr+N_transport+".txt"
                force_data = np.loadtxt("/home/hlab/bpbot/data/force/out.txt")
                force_data2 = np.insert(force_data, 0, np.insert(F0_reset, 0, 0))
                np.savetxt(recorded_path ,force_data2)

            else:
                print("After grasping: spin and transport")
                spin(vel=0.5)
                clear_motion([])        
        
        elif jnt != []:
            LIFT_STOPPED = True
            print("untangling")
            print("After grasping: fling")
            os.system("bash /home/hlab/bpbot/script/start_ft.sh")
            fling(j3=j4_range[N_transport], j4=j5_range[N_transport], vel=0.5)
            # fling(j3=30, j4=40, vel=0.5)
            clear_motion([])

            os.system("bash /home/hlab/bpbot/script/stop_ft.sh")
            print("Additional, we record the force in ")
            recorded_path = "/home/hlab/bpbot/data/force/raw_"+tstr+N_transport+".txt"
            force_data = np.loadtxt("/home/hlab/bpbot/data/force/out.txt")
            force_data2 = np.insert(force_data, 0, np.insert(F0_reset, 0, 0))
            np.savetxt(recorded_path ,force_data2)

            if PLAY:
                _, fout = monitoring(1, stop=False)
                # ft = np.mean(fout[:,-10:], axis=0)
                ft = np.mean(fs.filter(np.arange(len(fout)), fout, method="median", param=11)[1], axis=0)
                print("After fling value ", ft)
                if ft[2] < 0.2:
                    print("grasp nothing! ")
                    nxt.openHandToolLft()
                    nxt.openHandToolRgt()
                    nxt.setInitial(arm='all', tm=3)
                    break

        else:
            spin(vel=0.5)
            clear_motion([])        
        # check again 

        
        jnt, ft = transport()
        clear_motion(jnt)
        print("F/T output: ", ft)
        
        if jnt == [] and ft[2] <= k_fail + 0.1: 
            print("Adjust k_fail", k_fail, 'N -> ', end='')
            l_fail.append(ft[2])
            print("l_fail: ", l_fail)
            # k_fail = np.mean(l_fail)
            print(k_fail, "[N]")
            np.savetxt("/home/hlab/Desktop/ffail.txt", l_fail)

            put_ok()
            break
        elif ft[2] < 0.1:
            print("Grasp nothing")
            put_back()
        else:
            print("Transport failed, try again")
            N_transport += 1

        if  jnt != [] and LIFT_STOPPED != False:
            print("Here is timing to decrease k_stop: ", k_stop)
            # k_stop -= 0.1
            print(" -> ", k_stop)
        
        print("# Fling : ", N_transport)
        if N_transport >= 3:
            put_back()
            break
        
        if jnt == [] and ft[2] > k_fail: 
            print("Detected grasping two objects. put back and startover!!!!!")
            put_back()
            break

    print("Finish one picking attempt! ")
    i+=1

# ==================================== MAIN OVER =====================================