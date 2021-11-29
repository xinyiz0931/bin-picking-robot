import os
import sys
import math
import random
from numpy.core.numeric import full

from numpy.lib.type_check import _imag_dispatcher

# execute the script from the root directory etc. ~/src/myrobot
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import numpy as np
import configparser
import matplotlib.pyplot as plt
from datetime import datetime as dt

from example.binpicking import *
from grasping.graspability import Gripper, Graspability
from tangle_solution.topo_coor import LineDetection
from tangle_solution.entanglement_map import EntanglementMap

from utils.base_utils import *
from utils.transform_util import *
from utils.vision_utils import *

def main():
    
    # tunable parameter
    length_thre = 15
    distance_thre = 3
    sliding_size = 100
    sliding_stride = 25


    # root_dir = "D:\\code\\dataset\\drag_pick\\20211025121717"
    # img_path = os.path.join(root_dir, "depth1.png")
    img_path = "./vision/depth/depth0.png"

    img = cv2.imread(img_path)
    topo_img = cv2.resize(adjust_grayscale(img), (250,250))

    ld = LineDetection()
    lines_2d, lines_3d, lines_num, drawn = ld.detect_line(topo_img,length_thre, distance_thre,vis=True)

    em = EntanglementMap(length_thre, distance_thre, sliding_size, sliding_stride)
    emap, wmat_vis,w,d = em.entanglement_map(topo_img)
    norm_emap = adjust_array_range(emap, range=(0,255), if_img=True)

    # control group #1
    lmap = em.line_map(topo_img)
    bmap = em.brightness_map(topo_img)

    # grasp detection A
    # gripper = Gripper(finger_w=7,finger_h=12,gripper_width=30)
    gripper = Gripper(finger_w=5,finger_h=15,gripper_width=50)
    
    grasps, input_img, full_image = detect_grasp_point(gripper,5,img_path, (0,0,img.shape[0],img.shape[1]))
    drawn_gmap = gripper.draw_grasp(grasps[:5], img.copy())
    
    # drawc = draw_all_grasp(gripper, img_path, grasps, (0,0,img.shape[0],img.shape[1]))
    # get graspability map
    gmask = cv2.imread(f"./vision/tmp/gmap_{grasps[0][-2]}_{grasps[0][-1]}.png", 0)

    # rank grasps
    # for g in grasps:
    #     emap = cv2.resize(emap, (img.shape[0],img.shape[1]))
    #     g.append(emap[g[2]][g[1]])
    # grasps_array = np.array(grasps)
    # sorted_grasps = grasps_array[np.argsort(grasps_array[:, -1])]
    nontangle_grasps, input_img, full_image = detect_nontangle_grasp_point(gripper,10,img_path, (0,0,img.shape[0],img.shape[1]), norm_emap)
    drawn_emap = gripper.draw_grasp([nontangle_grasps[0]], img.copy())

    # visulization
    fig = plt.figure()

    fig.add_subplot(241)
    plt.imshow(img, cmap='gray')
    plt.title("depth image")

    fig.add_subplot(242)
    plt.imshow(drawn)
    plt.axis("off")
    plt.title("edges")


    fig.add_subplot(243)
    plt.imshow(img)
    plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.4, cmap='jet')
    plt.title("entanglement map with obj")

    fig.add_subplot(244)
    plt.imshow(cv2.resize(norm_emap, (img.shape[1], img.shape[0])), interpolation='bilinear', cmap='jet')
    plt.title("entanglement map")

    fig.add_subplot(245)
    plt.imshow(lmap, cmap='jet')
    for i in range(lmap.shape[1]):
        for j in range(lmap.shape[0]):
            text = plt.text(j, i, int(lmap[i,j]), ha="center", va="center", color="w")
    plt.title("line map")
    
    fig.add_subplot(246)
    plt.imshow(bmap, cmap='jet')
    for i in range(bmap.shape[1]):
        for j in range(bmap.shape[0]):
            text = plt.text(j, i, np.round(bmap[i, j],2), ha="center", va="center", color="w")
    plt.title("brightness map")
    
    # fig.add_subplot(247)
    # plt.imshow(emap, cmap='jet')
    # for i in range(emap.shape[1]):
    #     for j in range(emap.shape[0]):
    #         text = plt.text(j, i, np.round(emap[i, j],2),ha="center", va="center", color="w")

    # grasp related
    fig.add_subplot(247)
    plt.imshow(drawn_gmap)
    # plt.imshow(gmask, alpha=0.5,cmap='jet')
    plt.axis("off")
    plt.title("grasp by graspability map")
    

    fig.add_subplot(248)
    drawc = cv2.cvtColor(drawn_emap, cv2.COLOR_BGR2RGB)
    plt.imshow(drawn_emap)
    plt.axis("off")
    plt.title("grasp by entanglement map")
    
    plt.tight_layout()
    plt.get_current_fig_manager().full_screen_toggle()
    plt.show()

    fig2 = plt.figure() # for grasp process
    fig2.add_subplot(241)
    plt.imshow(cv2.imread(f"./vision/tmp/Wt_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0))
    plt.axis("off")

    fig2.add_subplot(242)
    plt.imshow(cv2.imread(f"./vision/tmp/Ht_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0))
    plt.axis("off")
    
    fig2.add_subplot(243)
    plt.imshow(cv2.imread(f"./vision/tmp/T_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0))
    plt.axis("off")

    fig2.add_subplot(244)
    plt.imshow(cv2.imread(f"./vision/tmp/T_Cbar_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0))
    plt.axis("off")
    # plt.imshow(img)
    # plt.imshow(cv2.resize(emap, (img.shape[1], img.shape[0])), interpolation='bilinear', alpha=0.5, cmap='jet')
    # plt.title("entanglement map with obj")


    fig2.add_subplot(245)
    plt.imshow(cv2.imread(f"./vision/tmp/Wc_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0))
    plt.axis("off")

    fig2.add_subplot(246)
    plt.imshow(cv2.imread(f"./vision/tmp/Hc_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0))
    plt.axis("off")

    fig2.add_subplot(247)
    plt.imshow(cv2.imread(f"./vision/tmp/C_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0))
    plt.axis("off")

    
    fig2.add_subplot(248)

    plt.imshow(drawn_emap)
    plt.imshow(cv2.imread(f"./vision/tmp/T_Cbar_{nontangle_grasps[0][-2]}_{nontangle_grasps[0][-1]}.png", 0),alpha=0.5, cmap='jet')
    plt.axis("off")

    plt.show()

    

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()
    
    main()

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))
