import os
import sys


sys.path.append("./")
from example.binpicking import *

def main():

    ROOT_DIR = os.path.abspath("./")
    img_path = os.path.join(ROOT_DIR, "vision/test/depth4.png")
    # img_path = "/media/xinyi/Files/code/dataset/labeled_pool/20210616045533_360_333_4_1_0.png"
    # ROOT_DIR = "C:\\Users\\matsumura\\Documents\\BinSimulator\\XYBin\\bin\\exp\\6DPOSE\\20211223182100"
    # img_path = os.path.join(ROOT_DIR, "depth.png")
    # ============== REAL WORLD SETUP ==============
    # finger_w=6.5
    # finger_h=20
    # open_w = 48
    # gripper_size = 250
    # ============== DEPTH SETUP (500x500)==============
    # finger_w=13
    # finger_h=40
    # open_w = 96/2 - 10
    # gripper_size = 500
    finger_w=3
    finger_h=10
    open_w = 20
    gripper_size = 500
    # ============== SIMBIM SETUP (225x225)==============
    # finger_w=5.85
    # finger_h=18
    # open_w = 43.2
    # gripper_size = 225

    rotation_step = 22.5
    depth_step = 50
    hand_depth = 50

    main_proc_print("Rotation step: {}".format(rotation_step))
    main_proc_print("Depth step: {}".format(depth_step))

    margins = (0,0,500,500)
    g_params = (rotation_step, depth_step, hand_depth)
    h_params = (finger_h, finger_w, open_w, gripper_size)

    grasps, input_img, full_image = detect_grasp_point(n_grasp=10, img_path=img_path, margins=margins, g_params=g_params, h_params=h_params)
    # grasps, input_img, full_image = detect_target_oriented_grasp(10, ROOT_DIR, margins, g_params, h_params)

    # temporal printing
    if grasps:
        result_print(f"Top grasp: pixel location=({grasps[0][1]},{grasps[0][2]}), angle={grasps[0][4]*180/math.pi}, width={grasps[0][-1]}")

    if grasps:
        for i in range(len(grasps)):
            result_print(f"Top grasp={grasps[i][0]}: pixel location=({grasps[i][1]},{grasps[i][2]}), angle={grasps[i][4]*180/math.pi}, width={grasps[i][-1]}")
if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
