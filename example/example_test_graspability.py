import os
from bpbot.binpicking import *

def main():

    root_dir = os.path.abspath("./")
    img_path = os.path.join(root_dir, "data/test/depth4.png")
    # ============== REAL WORLD SETUP ==============
    # finger_w=6.5
    # finger_h=20
    # open_w = 48
    # gripper_size = 250
    # ============== DEPTH SETUP (500x500)==============
    finger_w=13
    finger_h=40
    open_w = 96/2-10
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

    g_params = (rotation_step, depth_step, hand_depth)
    h_params = (finger_w, finger_h, open_w, gripper_size)

    grasps, img_input, img_grasp = detect_grasp_point(n_grasp=5, img_path=img_path, g_params=g_params, h_params=h_params)
    # grasps, input_img, full_image = detect_target_oriented_grasp(10, ROOT_DIR, margins, g_params, h_params)

    # temporal printing
    if grasps:
        result_print(f"Top grasp: pixel location=({grasps[0][1]},{grasps[0][2]}), angle={grasps[0][4]*180/math.pi}, width={grasps[0][-1]}")

    if grasps:
        for i in range(len(grasps)):
            result_print(f"Top grasp={grasps[i][0]}: pixel location=({grasps[i][1]},{grasps[i][2]}), angle={grasps[i][4]*180/math.pi}, width={grasps[i][-1]}")
    
    cv2.imshow("windows", img_grasp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
