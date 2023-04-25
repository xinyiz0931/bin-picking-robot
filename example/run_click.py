import os
import random
import importlib
spec = importlib.util.find_spec("cnoid")
FOUND_CNOID = spec is not None
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
from bpbot.utils import * 
import timeit
import numpy as np
start = timeit.default_timer()

# ========================== define path =============================
container = "pick"
LOG_ON = True

# get root dir
root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
print(f"[*] Start at {root_dir} ")

depth_dir = os.path.join(root_dir, "data/depth/")

img_path = os.path.join(root_dir, "data/depth/depth.png")
crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
config_path = os.path.join(root_dir, "cfg/config.yaml")
calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
mf_path = os.path.join(root_dir, "data/motion/motion.dat")
traj_path = os.path.join(root_dir, "data/motion/motion_ik.dat")
draw_path = os.path.join(root_dir, "data/depth/result.png")

# ======================= get config info ============================

cfg = BinConfig(config_path)
cfgdata = cfg.data

# ======================== get depth img =============================

point_array = capture_pc()
if point_array is None: 
    print("[!] Exit! ")
    sys.exit()
img, img_blur = pc2depth(point_array, cfgdata, container=container)
cv2.imwrite(img_path, img_blur)

crop = crop_roi(img_blur, cfgdata, container=container, bounding=False)
cv2.imwrite(crop_path, crop)

# =======================  compute grasp =============================
# grasps = [] 
# drawn = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
# def on_click(event,x,y,flags,param):
#     if event == cv2.EVENT_LBUTTONDOWN:
#         # cv2.circle(drawn,(x,y),5,(0,255,0),-1)
#         g = detect_grasp_orientation((x,y), crop_path, g_params=cfgdata["graspability"], h_params=cfgdata["hand"]["right"])
#         print(f"[{x},{y}] => {g}")
#         draw_grasp(g, drawn)
#         grasps.append(g)

# # p_clicked = [[102,107], [308, 243]]
# cv2.namedWindow("Click to select grasp")
# cv2.setMouseCallback("Click to select grasp", on_click)
# while(True):
#     cv2.imshow("Click to select grasp", drawn)
#     key = cv2.waitKey(1) & 0xFF
#     if key == ord('r'):
#         print("Refresh! ")
#         drawn = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
#         grasps = []
#     if key == ord('q') or key == 13:
#         cv2.destroyAllWindows()
#         break
# g_pick = grasps[-1]

grasps = detect_grasp(n_grasp=5, 
                            img_path=crop_path, 
                            g_params=cfgdata['graspability'],
                            h_params=cfgdata["hand"]["left"])

g_pick = grasps[0]
p_pick_tcp, g_pick_wrist = transform_image_to_robot(g_pick, point_array, cfgdata, hand="right", container="pick")

gen_motion_dynamic(mf_path, g_pick_wrist, orientation='v')
# gen_motion_pick(mf_path, g_pick_wrist)
print("[*] **Pick**! Grasp : (%d,%d,%.1f) -> Tcp : (%.3f,%.3f,%.3f)" % (*g_pick, *p_pick_tcp))

img_grasp = draw_grasp(g_pick, crop_path, cfgdata["hand"]["right"])
cv2.imwrite(draw_path, img_grasp)

# # =======================  generate motion ===========================
if FOUND_CNOID: 
    plan_success = load_motionfile(mf_path)
    print(f"[*] Motion planning succeed? ==> {plan_success}")
    if plan_success.count(True) == len(plan_success):
        nxt = NxtRobot(host='[::]:15005')
        motion_seq = get_motion()
        num_seq = int(len(motion_seq)/20)
        print(f"[*] Total {num_seq} motion sequences! ")
        motion_seq = np.reshape(motion_seq, (num_seq, 20))

        os.system("bash /home/hlab/bpbot/script/start_ft.sh")
        nxt.playMotionFT(motion_seq)
        os.system("bash /home/hlab/bpbot/script/stop_ft.sh")
        from bpbot.device import DynPick
        ft = DynPick()
        ft.plot_file()
        if LOG_ON:
            print("[*] Save results")
            tdatetime = dt.now()
            tstr = tdatetime.strftime('%Y%m%d%H%M%S')
            save_dir = "/home/hlab/Desktop/exp" 
            tstr_dir = os.path.join(save_dir, tstr)
            if not os.path.exists(save_dir): os.mkdir(save_dir)
            if not os.path.exists(tstr_dir): os.mkdir(tstr_dir)
            import shutil
            shutil.copyfile(crop_path, os.path.join(tstr_dir, "depth.png"))
            shutil.copyfile(draw_path, os.path.join(tstr_dir, "grasp.png"))
            shutil.copyfile(os.path.join(root_dir, "data/force/out.txt"), os.path.join(tstr_dir, "force.txt"))
            shutil.copyfile(os.path.join(root_dir, "data/force/out.png"), os.path.join(tstr_dir, "force.png"))
    else:
        print("[!] Motion planning failed ...")
    
    # nxt.playMotion(motion_seq)

# end = timeit.default_timer()
# print("[*] Time: {:.2f}s".format(end - start))
