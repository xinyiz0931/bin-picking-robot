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

def check_multi_view(root_dir, shape, proj_euler_angle = [0, 0, 0]):

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

    """compute tangleship"""
    graph = tc6d.make_sim_graph(template, pose, cube1_pos)
    num_obj = len(graph)

    """visualize all objects and tangleship"""
    fig = plt.figure(figsize=(15, 9))
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.view_init(167, -87)

    # plot axis
    ax3d.plot([0, 10], [0, 0], [0, 0], color='red') # x
    ax3d.plot([0, 0], [0, 10], [0, 0], color='green') # y
    ax3d.plot([0, 1], [0, 0], [0, 10], color='blue') # z
    ax3d.set_xlim3d(-100, 100)
    ax3d.set_ylim3d(-100, 100)
    ax3d.set_zlim3d(-100, 100)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    plt.title("3D coordinate")

    sorted_index = list(range(num_obj))
    cmap = get_cmap(len(sorted_index))
    ax2 = fig.add_subplot(122)
    plt.xlim(-120, 120)
    plt.ylim(120, -120)

    labels = tc6d.compute_tangleship_with_projection(root_dir, graph, pose, proj_euler_angle)

    """read and display point cloud """
    # xyz = process_raw_pc(pc_path)
    # for p in xyz:
    #     ax3d.scatter(p[0], p[1], p[2], marker='^', alpha=0.3)

    for i, j in zip(sorted_index, range(num_obj)):
        node = graph[i]
        node_cmap = cmap(j)
        tc6d.draw_node(node, ax3d, alpha=1, color=node_cmap)
        # tc6d.draw_projection_node_new(node, proj_euler_angle , ax3d, alpha=0.4, color=node_cmap)
        tc6d.draw_projection_node_2d(node, proj_euler_angle , ax2, alpha=1, color=node_cmap)

    # for c in crossings:
    #     ax3d.scatter(c[0], 0, c[1], color='black')
    
    legend_elements = []
    for i in range(num_obj):
        legend_elements.append(Line2D([0], [0], color=cmap(i), lw=4, label=f'Object {i}'))
    ax3d.legend(handles=legend_elements, loc='center left')

    return labels

def check_result(labels):

    writhe = []
    pick_idx = -1
    for l in labels:
        writhe.append(len(labels.get(l)))
        if len(labels.get(l)) == 0:
            pick_idx = l
            result_print(f"Obj {pick_idx}: singulated -> {labels.get(l)}")
        elif all(c > 0 for c in labels.get(l)): 
            
            pick_idx = l
            result_print(f"Obj {pick_idx}: untangled -> {labels.get(l)}")

    # result_print(f"Labels: {labels}")
    result_print(f"Writhe: {writhe}")

    return pick_idx

def main():

    # root_dir = "C:\\Users\\matsumura\\Documents\\BinSimulator\\XYBin\\bin\\exp\\6DPOSE\\20211203223808"
    root_dir = "D:\\code\\dataset\\tangle_in_sim\\twist"
    shape = "c"

    phi_xy = 60
    phi_z = 45


    labels = check_multi_view(root_dir, shape)
    pick_idx = check_result(labels)
    # if pick_idx == -1:
    #     warning_print("Oops! No graspable object! ")
    #     for x in np.arange(0,91,phi_xy):
    #         for z in np.arange(0,360,phi_z):
    #             if x != 0:
    #                 view_direction = [x,0,z] # euler angle 
    #                 labels = check_multi_view(root_dir, shape, proj_euler_angle=view_direction)
    #                 check_result(labels)
                    # plt.show()

def main2():
    
    tc6d = TopoCoor6D()

    """Configurations defined by users"""
    # root_dir = "D:\\code\\dataset\\tangle_in_sim\\twist"
    root_dir = "C:\\Users\\matsumura\\Documents\\BinSimulator\\XYBin\\bin\\exp\\6DPOSE\\20211203223808"
    shape = "scylinder"

    # graph = []

    # pc_path = os.path.join(root_dir, "point.ply")
    # im_path = os.path.join(root_dir, "depth.png")
    # pose_path = os.path.join(root_dir, "pose.txt")

    # sk_path = f"./objmodel/skeleton_{shape}.json"
    # cd_path = f"./objmodel/collision_{shape}.txt"

    # # pcd = o3d.io.read_point_cloud(pc_path)
    # # o3d.visualization.draw_geometries([pcd])



    # with open(cd_path) as file:
    #     vhacd = file.readlines()

    # center = np.array([float(p) for p in vhacd[0].split(' ')])
    # cube1_pos = np.array([float(p) for p in vhacd[3].split(' ')])

    # pose = np.loadtxt(pose_path)

    # tok = TangleObjSke()
    # obj_ske = tok.load_obj(sk_path)
    
    # template = np.array(obj_ske["node"])

    # """compute tangleship"""
    # graph = tc6d.make_sim_graph(template, pose, center, cube1_pos)
    # num_obj = len(graph)

    # writhe_collection, height_collection = tc6d.compute_tangleship(root_dir, graph, pose)
    # result_print(f"Writhe: {np.round(writhe_collection, 3)}")
    # result_print(f"Height: {np.round(height_collection, 3)}")

    # writhe_mat = tc6d.compute_tangleship(root_dir, graph, pose)
    # voting = tc6d.compute_tangleship(root_dir, graph, pose)


    """initial coding for view directions sampling"""


    # view directions sampling
    # k = 0
    # for raw, yall in itertools.product([-90,0,90], repeat=2):
    #     fig = plt.figure(figsize=(15, 9))
    #     ax3d = fig.add_subplot(121, projection='3d')
    #     ax3d.view_init(167, -87)
    #     # ax3d.view_init(180,270)

    #     # plot axis
    #     ax3d.plot([0, 10], [0, 0], [0, 0], color='red') # x
    #     ax3d.plot([0, 0], [0, 10], [0, 0], color='green') # y
    #     ax3d.plot([0, 1], [0, 0], [0, 10], color='blue') # z
    #     ax3d.set_xlim3d(-100, 100)
    #     ax3d.set_ylim3d(-100, 100)
    #     ax3d.set_zlim3d(-100, 100)
    #     ax3d.set_xticklabels([])
    #     ax3d.set_yticklabels([])
    #     ax3d.set_zticklabels([])
    #     plt.title("3D coordinate")

    #     sorted_index = list(range(num_obj))
    #     cmap = get_cmap(len(sorted_index))
    #     ax2 = fig.add_subplot(122)
    #     plt.xlim(-100, 100)
    #     plt.ylim(100, -100)
    #     proj_euler_angle = [raw, 0, yall]
    #     main_proc_print(f"Calculate for {proj_euler_angle}")
        
    #     voting = tc6d.compute_tangleship_with_projection(root_dir, graph, pose, proj_euler_angle)
    #     for i, j in zip(sorted_index, range(num_obj)):
    #         node = graph[i]
    #         node_cmap = cmap(j)
    #         tc6d.draw_node(node, ax3d, alpha=1, color=node_cmap)
    #         # tc6d.draw_projection_node_new(node, proj_euler_angle , ax3d, alpha=0.4, color=node_cmap)
    #         tc6d.draw_projection_node_2d(node, proj_euler_angle , ax2, alpha=1, color=node_cmap)

    #     legend_elements = []
    #     for i in range(num_obj):
    #         legend_elements.append(Line2D([0], [0], color=cmap(i), lw=4, label=f'Object {i}'))
    #     # ax1 = fig.add_subplot(132)
    #     ax3d.legend(handles=legend_elements, loc='center left')
    #     # plt.axis('off')
    #     # vd = np.dot(rpy2mat(proj_euler_angle),[0,50,0])
    #     # ax3d.plot([0,vd[0]], [100,0],[0,vd[2]], alpha=1, color='red', marker='^')
    #     # if voting.count(0) != len(voting):
    #     #     plt.show()
    #     fig.savefig(f"image{k}.png")
    #     k += 1
    # print(f"total {k}")

    """visualize all objects and tangleship"""
    fig = plt.figure(figsize=(15, 9))
    ax3d = fig.add_subplot(121, projection='3d')
    ax3d.view_init(167, -87)
    # ax3d.view_init(180,270)

    # plot axis
    ax3d.plot([0, 10], [0, 0], [0, 0], color='red') # x
    ax3d.plot([0, 0], [0, 10], [0, 0], color='green') # y
    ax3d.plot([0, 1], [0, 0], [0, 10], color='blue') # z
    ax3d.set_xlim3d(-100, 100)
    ax3d.set_ylim3d(-100, 100)
    ax3d.set_zlim3d(-100, 100)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    plt.title("3D coordinate")

    sorted_index = list(range(num_obj))
    cmap = get_cmap(len(sorted_index))
    ax2 = fig.add_subplot(122)
    plt.xlim(-120, 120)
    plt.ylim(120, -120)

    # normal testing
    # proj_euler_angle = [0, 0, 45]
    proj_euler_angle = [0, 0, 0] # n_0
    labels = tc6d.compute_tangleship_with_projection(root_dir, graph, pose, proj_euler_angle)

    
    # revised to clearly see writhe
    writhe = []
        
    pick_idx = -1

    for l in labels:
        writhe.append(len(labels.get(l)))
        if len(labels.get(l)) == 0:
            pick_idx = l
            result_print(f"Obj {pick_idx}: singulated -> {labels.get(l)}")
        elif all(c > 0 for c in labels.get(l)): 
            
            pick_idx = l
            result_print(f"Obj {pick_idx}: untangled -> {labels.get(l)}")
    if pick_idx == -1:
        warning_print("Oops! No graspable object! ")

    result_print(f"Labels: {labels}")
    result_print(f"Writhe: {writhe}")

    """Solve for drag-picking"""
    phi_xy=90
    phi_z = 45

    for x in np.arange(0,91,phi_xy):
        for z in np.arange(0,360,phi_z):
            if x != 0:
                # euler angle
                print([x,0,z])
                # rot = np.dot(rpy2mat([x,0,z]), init_view)
                # rot =  rot / np.linalg.norm(rot)
                # print(np.degrees(rot))
                
    """read and display point cloud """
    # xyz = process_raw_pc(pc_path)
    # for p in xyz:
    #     ax3d.scatter(p[0], p[1], p[2], marker='^', alpha=0.3)

    for i, j in zip(sorted_index, range(num_obj)):
        node = graph[i]
        node_cmap = cmap(j)
        tc6d.draw_node(node, ax3d, alpha=1, color=node_cmap)
        # tc6d.draw_projection_node_new(node, proj_euler_angle , ax3d, alpha=0.4, color=node_cmap)
        tc6d.draw_projection_node_2d(node, proj_euler_angle , ax2, alpha=1, color=node_cmap)

    # for c in crossings:
    #     ax3d.scatter(c[0], 0, c[1], color='black')
    
    legend_elements = []
    for i in range(num_obj):
        legend_elements.append(Line2D([0], [0], color=cmap(i), lw=4, label=f'Object {i}'))
    # ax1 = fig.add_subplot(132)
    ax3d.legend(handles=legend_elements, loc='center left')
    # plt.axis('off')

    # visualize all graspable objects using depth image
    im = cv2.imread(im_path, 0)

    # ax3 = fig.add_subplot(133)
    # rot90 = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # ax3.imshow(rot90, cmap='gray')
    # plt.title("Depth image")
    plt.show()
    # fig = plt.figure()
    # for i in range(num_obj):
    #     if labels[i]==[]:
    #         # singulated
    #         mask_path = os.path.join(root_dir, f"mask_{i}.png")
    #         plt.imshow(im, cmap='gray')
    #         plt.imshow(cv2.imread(mask_path, 0), cmap='jet', alpha=0.4)
    #         plt.axis('off')
    #         plt.show()
    #     if labels[i].count(1) == len(labels[i]):
    #         # top
    #         mask_path = os.path.join(root_dir, f"mask_{i}.png")
    #         plt.imshow(im, cmap='gray')
    #         plt.imshow(cv2.imread(mask_path, 0), cmap='jet', alpha=0.4)
    #         plt.axis('off')
    #         plt.show()

    # mask_path = os.path.join(root_dir, "mask_3.png")
    # mask_path2 = os.path.join(root_dir, "mask_5.png")
    # plt.imshow(im, cmap='gray')
    # plt.imshow(cv2.imread(mask_path, 0) + cv2.imread(mask_path2, 0), cmap='jet', alpha=0.4)
    # # plt.imshow(cv2.imread(mask_path2, 0), cmap='jet', alpha=0.1)
    # plt.axis('off')
    # plt.show()


    


if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    main()
    plt.show()
    

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

