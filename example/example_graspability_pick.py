import os
from bpbot.binpicking import *
from bpbot import BinConfig

def main():

    root_dir = os.path.abspath("./")
    img_path = os.path.join(root_dir, "data/test/depth4.png")
    img_path = os.path.join(root_dir, "./roi.png")
    config_path = os.path.join(root_dir, "cfg/config.yaml")

    bincfg = BinConfig(config_path)
    cfg = bincfg.data

    # ============== REAL WORLD SETUP ==============
    # finger_w=6.5
    # finger_h=20
    # open_w = 48
    # tplt_size = 250
    # ============== DEPTH SETUP (500x500)==============
    h_params = {"finger_length": 30,
                "finger_width":  12, 
                "open_width":    40}
    # h_params = cfg["hand"] 
    g_params = {"rotation_step": 22.5, 
                "depth_step":    50,
                "hand_depth":    50}
    # g_params = cfg["graspability"]

    # ============== SIMBIM SETUP (225x225)==============
    # finger_w=5.85
    # finger_h=18
    # open_w = 43.2
    # tplt_size = 225

    print("[*] Hand params: {}".format(h_params))
    print("[*] FGE params: {}".format(g_params))

    # g_params = (rotation_step, depth_step, hand_depth)
    # h_params = (finger_w, finger_h, open_w, tplt_size)
    # g_params = cfg.g_params
    # h_params = cfg.h_params

    grasps = detect_grasp(n_grasp=5, img_path=img_path, 
                            g_params=g_params,
                            h_params=h_params)
    img = cv2.imread(img_path)
    img_grasp = draw_grasp(grasps, img, h_params)
    # grasps, img_input, img_grasp = detect_grasp(n_grasp=5, img_path=img_path, g_params=g_params, h_params=h_params)
    # grasps, input_img, full_image = detect_target_oriented_grasp(10, ROOT_DIR, margins, g_params, h_params)

    # temporal printing

    if grasps is not None:
        for i in range(len(grasps)):
            print(f"[*] #{i}: pixel=({grasps[i][0]},{grasps[i][1]}), angle={grasps[i][2]}")
    cv2.imshow("windows", img_grasp)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    print("[*] Time: {:.2f}s".format(end - start))

    
