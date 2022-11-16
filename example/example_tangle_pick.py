import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from bpbot.binpicking import *
from bpbot.config import BinConfig

def main():
    h_params = {
        "finger_height": 40,
        "finger_width":  13, 
        "open_width":    50
    }
    g_params = {
        "rotation_step": 22.5, 
        "depth_step":    50,
        "hand_depth":    50
    }
    t_params = {
        "compressed_size": 250,
        "len_thld": 15,
        "dist_thld": 3,
        "sliding_size": 125,
        "sliding_stride": 25
    }
    
    bincfg = BinConfig()
    cfg = bincfg.data
    # img_path = os.path.join("./data", "test", "depth4.png")

    img_path = "C:\\Users\\xinyi\\Documents\\XYBin_Pick\\bin\\tmp\\depth.png"
    img = cv2.imread(img_path)
    img_edge, num_edge = detect_edge(img, t_params)

    emap = get_entanglement_map(img, t_params)

    grasps= detect_grasp(n_grasp=10, img_path=img_path, 
                                    g_params=g_params, 
                                    h_params=h_params)
    img_grasp = draw_grasp(grasps, img.copy())

    t_grasps = detect_nontangle_grasp(n_grasp=10, img_path=img_path, 
                                    g_params=g_params, 
                                    h_params=h_params,
                                    t_params=t_params)
                                    

    t_img_grasp = draw_grasp(t_grasps, img.copy())
    # # visulization
    fig = plt.figure()

    fig.add_subplot(231)
    plt.imshow(img, cmap='gray')
    plt.title("Depth")

    fig.add_subplot(232)
    plt.imshow(img_edge)
    plt.axis("off")
    plt.title("Edge")

    fig.add_subplot(233)
    plt.imshow(img)
    plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.4, cmap='jet')
    plt.title("Depth + EMap")

    fig.add_subplot(234)
    plt.imshow(img_grasp)
    plt.axis("off")
    plt.title("FGE")

    fig.add_subplot(235)
    plt.imshow(t_img_grasp)
    plt.axis("off")
    plt.title("FGE + EMap")

    # grasp related
    fig.add_subplot(236)
    plt.imshow(t_img_grasp)
    plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.3, cmap='jet')
    plt.axis("off")
    plt.title("FGE + EMap (EMap shown)")
    
    plt.tight_layout()
    plt.get_current_fig_manager().full_screen_toggle()
    plt.savefig("./test.png")
    plt.show()


if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()
    
    main()

    end = timeit.default_timer()
    print("[*] Time: {:.2f}s".format(end - start))
