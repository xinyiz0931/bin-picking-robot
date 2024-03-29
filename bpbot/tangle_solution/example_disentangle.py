import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from bpbot.utils import * 
from bpbot.tangle_solution import TopoCoor6D

def load(tc6d, graph, proj_ea=[0,0,0]):
    [xz, y, _] = proj_ea

    proj_view_ea = [xz,-y,0] # 'xyz' seq
    proj_obj_ea = [y,-xz,0] # 'yzx' seq
    obj_num = len(graph)
    """visualize all objects and tangleship"""
    fig = plt.figure(figsize=(7, 7))
    ax2 = fig.add_subplot(111)

    crossings = tc6d.compute_tangleship(graph, proj_obj_ea)
    
    # visualize the objects
    cmap = get_cmap(obj_num)
    cmap_list = [
                 (145/255,174/255,253/255,1),
                 (51/255,102/255,255/255,1),
                 (190/255, 190/255,190/255,1),
                 (244/255,89/255,144/255,1),
                 (244/255,89/255,144/255,1)]                 
    print(np.round(cmap_list, 3))
    # cmap_list = [
    #              (190/255, 190/255,190/255,1),
    #              (244/255,89/255,144/255,1)
    # ]

    for i, j in zip(range(obj_num), range(obj_num)):
        node = graph[i]
        # node_cmap = cmap(j)
        node_cmap = cmap_list[j]
        tc6d.draw_node_2d(node, ax2, alpha=1, color=node_cmap, proj_ea=proj_obj_ea)
    
    # tc6d.draw_node_2d(graph[1], ax2, alpha=1, color=cmap_list[1], proj_ea=proj_obj_ea)
    # tc6d.draw_node_2d(graph[0], ax2, alpha=1, color=cmap_list[0], proj_ea=proj_obj_ea)
    # count for under-crossings
    undercrossings = []
    for i in range(obj_num):
        under_num = 0
        for k, l in enumerate(crossings["label"][i]):
            # check for number of undercrossing
            if l == -1:
                under_num += 1
                cpoint = crossings["point"][i][k]
                other_index = crossings["obj"][i][k]
                # ax2.scatter(cpoint[0],cpoint[2],color=cmap(other_index),zorder=2)
                ax2.scatter(cpoint[0],cpoint[2],color=cmap_list[other_index],zorder=2, s=300, edgecolors='white')
    
    
        undercrossings.append(under_num)
    min_under_idx = undercrossings.index(min(undercrossings))
    print(f"[*] Grasp object index: {min_under_idx}")
    legend_elements = []
    for i in range(obj_num):
        # legend_elements.append(Line2D([0], [0], color=cmap(i), lw=4, label=f'Object {i}'))
        legend_elements.append(Line2D([0], [0], color=cmap_list[i], lw=4, label=f'Object {i}'))
    ax2.set_xlim(-112.5, 112.5)
    ax2.set_ylim(112.5, -112.5)
    plt.show()


def view(tc6d, pose, proj_ea=[0,0,0]):

    [xz,y,_] = proj_ea # z=0 here
    proj_view_ea = [xz,-y,0] # 'xyz' seq
    proj_obj_ea = [y,-xz,0] # 'yzx' seq

    """compute tangleship"""
    graph = tc6d.create_sim_graph(pose)
    obj_num = len(graph)

    """visualize all objects and tangleship"""
    fig = plt.figure(figsize=(15, 7))
    ax3d = fig.add_subplot(121, projection='3d')
    ax2 = fig.add_subplot(122)

    crossings = tc6d.compute_tangleship(graph, proj_obj_ea)
    
    # visualize the objects
    cmap = get_cmap(obj_num)
    for i, j in zip(range(obj_num), range(obj_num)):
        node = graph[i]
        node_cmap = cmap(j)
        tc6d.draw_node_3d(node, ax3d, alpha=1, color=node_cmap)
        tc6d.draw_node_2d(node, ax2, alpha=1, color=node_cmap, proj_ea=proj_obj_ea)
    
    # count for under-crossings
    undercrossings = []
    for i in range(obj_num):
        under_num = 0
        for k, l in enumerate(crossings["label"][i]):
            # check for number of undercrossing
            if l == -1:
                under_num += 1
                cpoint = crossings["point"][i][k]
                other_index = crossings["obj"][i][k]
                ax2.scatter(cpoint[0],cpoint[2],color=cmap(other_index),zorder=2)
        undercrossings.append(under_num)
    min_under_idx = undercrossings.index(min(undercrossings))
    print(f"[*] Grasp object index: {min_under_idx}")

    # plot the view direction
    viewpoint = np.dot(rpy2mat(proj_view_ea), [0,1,0])
    viewpoint =  viewpoint / np.linalg.norm(viewpoint) * 50
    # start point (graph[min_under_idx][0][0], graph[min_under_idx][0][1], graph[min_under_idx][0][2])
    ax3d.quiver(0, 0, 0, viewpoint[0], viewpoint[1], viewpoint[2], 
                length=2, color='black', alpha=0.5, lw=2, arrow_length_ratio=0.25)
    
    # setups for figure
    ax3d.plot([0, 10], [0, 0], [0, 0], color='red') # x
    ax3d.plot([0, 0], [0, 10], [0, 0], color='green') # y
    ax3d.plot([0, 0], [0, 0], [0, 10], color='blue') # z
    ax3d.set_xlim3d(-112.5, 112.5)
    ax3d.set_ylim3d(-100, 100)
    ax3d.set_zlim3d(-112.5, 112.5)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    ax3d.set_box_aspect(aspect = (1,1,1))
    # ax3d.view_init(180, 270) # top view
    ax3d.view_init(145, -90)
    plt.title("3D coordinate")
    ax2.set_xlim(-112.5, 112.5)
    ax2.set_ylim(112.5, -112.5)
    plt.title("2D view")

    legend_elements = []
    for i in range(obj_num):
        legend_elements.append(Line2D([0], [0], color=cmap(i), lw=4, label=f'Object {i}'))
    ax3d.legend(handles=legend_elements, loc='center left')

    plt.show()

def main():

    root_dir =  "C:\\Users\\xinyi\\Documents\\XYBin_Collected\\example_u"
    pose_path = os.path.join(root_dir, "pose.txt")
    node_path = os.path.join(root_dir, "node.txt")
    shape = "u"
    phi_xz = 20 # [0, 90]
    phi_y = 45 # [0, 360]


    sk_path = f"./objmodel/skeleton_{shape}.json"
    cd_path = f"./objmodel/collision_{shape}.txt"
    
    tc6d = TopoCoor6D(sk_path, cd_path)

    views = tc6d.sample_view(phi_xz, phi_y)
    print(views)
    
    # pose = np.loadtxt(pose_path)
    # view(tc6d, pose)
    graph = np.loadtxt(node_path, delimiter=',')
    graph = np.reshape(graph, (5, 9, 3))
    load(tc6d, graph, proj_ea=[0,0,0])
    graph = graph[:2]
    load(tc6d, graph, proj_ea=[0,0,0])
    # load(tc6d, graph, proj_ea=[70,20,0])
    # load(tc6d, graph, proj_ea=[45,10,0])
    # for v in views:
    #     load(tc6d, graph, proj_ea=v)

if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    main()

    end = timeit.default_timer()
    print("[*] Time: {:.2f}s".format(end - start))