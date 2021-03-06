import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from bpbot.binpicking import *
from bpbot.config import BinConfig

def main():
    root_dir = os.path.abspath("./")
    config_path = os.path.join(root_dir, "cfg/config.yaml")

    bincfg = BinConfig(config_path)
    cfg = bincfg.config
    img_path = os.path.join("./data", "test", "depth4.png")
    # img_path = os.path.join("./data", "depth", "depth_cropped_2.png") 

    img_edge, num_edge = detect_edge(img_path, cfg["tangle"])

    _, emap = get_entanglement_map(img_path, cfg["tangle"])

    
    grasps, img_input= detect_grasp_point(n_grasp=10, 
                                                    img_path=img_path, 
                                                    g_params=cfg["graspability"], 
                                                    h_params=cfg["hand"])
    img_grasp = draw_grasps(grasps, img_path, cfg["hand"])


    t_grasps, t_img_input = detect_nontangle_grasp_point(n_grasp=10, 
                                                        img_path=img_path, 
                                                        g_params=cfg["graspability"], 
                                                        h_params=cfg["hand"],
                                                        t_params=cfg["tangle"])
    t_img_grasp = draw_grasps(t_grasps, img_path, cfg["hand"])

    # # visulization
    fig = plt.figure()

    fig.add_subplot(231)
    plt.imshow(img_input, cmap='gray')
    plt.title("Depth")

    fig.add_subplot(232)
    plt.imshow(img_edge)
    plt.axis("off")
    plt.title("Edge")

    fig.add_subplot(233)
    plt.imshow(img_input)
    plt.imshow(cv2.resize(emap, (img_input.shape[1], img_input.shape[0])), interpolation='bilinear', alpha=0.4, cmap='jet')
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
    plt.imshow(cv2.resize(emap, (img_input.shape[1], img_input.shape[0])), interpolation='bilinear', alpha=0.3, cmap='jet')
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
    main_proc_print("Time: {:.2f}s".format(end - start))
