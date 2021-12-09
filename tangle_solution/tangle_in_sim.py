import os
import sys
sys.path.append("./")
import random

import numpy as np
from scipy import ndimage
import cv2
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.lines import Line2D
from scipy.spatial.transform import Rotation as R
import itertools
import glob
from utils.plot_utils import *
from utils.base_utils import *
from utils.transform_utils import *
from utils.vision_utils import *
from tangle_solution.topo_coor_6d import TopoCoor6D
from tangle_obj_skeleton import TangleObjSke

def check_multi_view(root_dir, shape, euler_angle = [0, 0, 0]):

    main_proc_print("-----------------------------------------------")
    tc6d = TopoCoor6D()
    graph = []

    pc_path = os.path.join(root_dir, "point.ply")
    im_path = os.path.join(root_dir, "depth.png")
    pose_path = os.path.join(root_dir, "pose.txt")

    sk_path = f"./objmodel/skeleton_{shape}.json"
    cd_path = f"./objmodel/collision_{shape}.txt"

    with open(cd_path) as file:
        vhacd = file.readlines()

    center = np.array([float(p) for p in vhacd[0].split(' ')])
    cube1_pos = np.array([float(p) for p in vhacd[3].split(' ')])
    pose = np.loadtxt(pose_path)
    tok = TangleObjSke()
    obj_ske = tok.load_obj(sk_path)
    template = np.array(obj_ske["node"])

    # proj_obj: yxz ===> proj_view:xyz 
    [xz,y,_] = euler_angle # z=0 here

    proj_view_ea = [xz,-y,0] # 'xyz' seq

    proj_obj_ea = [y,-xz,0] # 'yzx' seq

    """compute tangleship"""
    graph = tc6d.make_sim_graph(template, pose, cube1_pos)
    graph = np.array(graph)

    # graph -= np.array([0, np.min(graph[:,1]),0])

    num_obj = len(graph)

    """visualize all objects and tangleship"""
    fig = plt.figure(figsize=(15, 7))
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.view_init(167, -87)

    # plot axis
    ax3d.plot([0, 10], [0, 0], [0, 0], color='red') # x
    ax3d.plot([0, 0], [0, 10], [0, 0], color='green') # y
    ax3d.plot([0, 0], [0, 0], [0, 10], color='blue') # z
    ax3d.set_xlim3d(-100, 100)
    ax3d.set_ylim3d(-100, 100)
    ax3d.set_zlim3d(-100, 100)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_box_aspect(aspect = (1,1,1))
    plt.title("3D coordinate")

    sorted_index = list(range(num_obj))
    cmap = get_cmap(len(sorted_index))

    # ax2 = fig.add_subplot(122, projection='3d')
    # ax2.view_init(167, -87)

    # # plot axis
    # ax2.plot([0, 10], [0, 0], [0, 0], color='red') # x
    # ax2.plot([0, 0], [0, 10], [0, 0], color='green') # y
    # ax2.plot([0, 0], [0, 0], [0, 10], color='blue') # z
    # ax2.set_xlim3d(-100, 100)
    # ax2.set_ylim3d(-100, 100)
    # ax2.set_zlim3d(-100, 100)
    # # ax2.set_xticklabels([])
    # # ax2.set_yticklabels([])
    # # ax2.set_zticklabels([])

    # ax2.set_box_aspect(aspect = (1,1,1))

    ax2 = fig.add_subplot(122)
    # ax2.plot([0, 10], [0, 0], color='red') # x
    # ax2.plot([0, 0], [0, 0], color='blue') # z
    # ax2.scatter(0,0,0, color='blue')
    plt.xlim(-120, 120)
    plt.ylim(120, -120)
    # plt.xlim(-73, 20)
    # plt.ylim(-33, 2)

    # yzx rotation
    labels = tc6d.compute_tangleship_with_projection(root_dir, graph, pose, proj_obj_ea)

    """read and display point cloud """
    # xyz = process_raw_pc(pc_path)
    # for p in xyz:
    #     ax3d.scatter(p[0], p[1], p[2], marker='^', alpha=0.3)

    for i, j in zip(sorted_index, range(num_obj)):
        node = graph[i]
        node_cmap = cmap(j)
        tc6d.draw_node(node, ax3d, alpha=0.2, color=node_cmap)
        # tc6d.draw_projection_node_new(node, proj_euler_angle , ax3d, alpha=0.4, color=node_cmap)
        tc6d.draw_projection_node_2d(node, proj_obj_ea , ax2, alpha=0.2, color=node_cmap)
        # tc6d.draw_projection_node_3d(node, proj_obj_ea , ax2, alpha=0.2, color=node_cmap)

    # for c in crossings:
    #     ax3d.scatter(c[0], 0, c[1], color='black')
    
    legend_elements = []
    for i in range(num_obj):
        legend_elements.append(Line2D([0], [0], color=cmap(i), lw=4, label=f'Object {i}'))
    ax3d.legend(handles=legend_elements, loc='center left')

    writhe = []
    pick_idx = -1
    # pick_idx_col = []
    max_height = -99999
    for l in labels:
        writhe.append(len(labels.get(l)))
        if len(labels.get(l)) == 0 or all(c > 0 for c in labels.get(l)):
            result_print(f"checking obj {l} height -> {tc6d.get_proj_height(graph[l], proj_obj_ea)}! ")
            if tc6d.get_proj_height(graph[l], proj_obj_ea) > max_height:
                pick_idx = l
                max_height = tc6d.get_proj_height(graph[l], proj_obj_ea)
                
    result_print(f"Target obj: {pick_idx}")

    rot = np.dot(rpy2mat(proj_view_ea), [0,1,0])
    rot =  rot / np.linalg.norm(rot)*80
    # ax3d.quiver(0,0,0, rot[0], rot[1], rot[2], length = 2, color='black', alpha=0.5,lw=2)
    # pick_idx = check_result(labels)
    if pick_idx != -1:
        tc6d.draw_node(graph[pick_idx], ax3d, alpha=1, color=cmap(pick_idx))
        # tc6d.draw_projection_node_3d(graph[pick_idx], proj_obj_ea , ax2, alpha=1, color=cmap(pick_idx))
        tc6d.draw_projection_node_2d(graph[pick_idx], proj_obj_ea , ax2, alpha=1, color=cmap(pick_idx))
        ax3d.quiver(graph[pick_idx][0][0], graph[pick_idx][0][1], graph[pick_idx][0][2], rot[0], rot[1], rot[2], length = 2, color='black', alpha=0.5,lw=2)
    
    # plt.savefig(f"./vision/tmp/{pick_idx}_{euler_angle[0]}_{euler_angle[1]}_{euler_angle[2]}.png")
    plt.show()
    return pick_idx
    # return labels

def check_result(labels):

    writhe = []
    pick_idx = -1
    pick_idx_col = []
    for l in labels:
        writhe.append(len(labels.get(l)))
        if len(labels.get(l)) == 0:
            # pick_idx = l
            pick_idx_col.append(l)
            result_print(f"Obj {pick_idx}: singulated -> {labels.get(l)}")
        elif all(c > 0 for c in labels.get(l)): 
            # pick_idx = l
            pick_idx_col.append(l)
            result_print(f"Obj {pick_idx}: untangled -> {labels.get(l)}")
            
    result_print(f"Writhe: {writhe}")

    return pick_idx

def main():

    root_dir = "C:\\Users\\matsumura\\Documents\\BinSimulator\\XYBin\\bin\\exp\\6DPOSE\\20211206210027"
    # root_dir = "D:\\code\\dataset\\tangle_in_sim\\20211205114225"
    shape = "scylinder"

    phi_xz = 60
    phi_y = 90

    tc6d = TopoCoor6D()
    pick_idx = check_multi_view(root_dir, shape)
    if pick_idx == -1:
        warning_print("Oops! No graspable object! ")
        for xz in np.arange(0,91,phi_xz):
            for y in np.arange(0,360,phi_y):
                if xz != 0:
                    # euler angle in 'xyz' seq
                    view_direction = [xz,y,0] 
                    proj_use = [y,-xz,0] # 'yxz' seq
                    print(f"------------ {view_direction} ==> {proj_use}--------------")
                    # labels = check_multi_view(root_dir, shape, euler_angle=view_direction)

if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    main()
    

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

