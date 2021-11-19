import os
import sys
sys.path.append("./")
import random
import open3d as o3d
import numpy as np
from scipy import ndimage
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans
from scipy.spatial.transform import Rotation as R
import itertools
import glob
from utils.plot_utils import *
from utils.base_utils import *
from tangle_solution.topo_coor_6d import TopoCoor6D
from tangle_obj_skeleton import TangleObjSke

def read_model():
    objmodel_dir = "./objmodel"

    obj1_path = os.path.join(objmodel_dir, "skeleton_ushape.txt")
    obj2_path = os.path.join(objmodel_dir, "skeleton_sshape.txt")

    node1 = np.loadtxt(obj1_path, delimiter=',')
    node2 = np.loadtxt(obj2_path, delimiter=',')

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plot_skeleton(node1, ax, (255,0,0))
    plot_skeleton(node2, ax, (0,255,0))
    plt.show()

    tc = TopoCoor()

    line1 = np.array([node2edge(node1)])
    line2 = np.array([node2edge(node2)])
    wmat, w, d = tc.topo_coor_from_two_edges(line1, line2)

    result_print("w: {:.5}, d: {:.5}".format(w,d))

def main():
    
    tc6d = TopoCoor6D()

    """Configurations defined by users"""

    root_dir = "C:\\Users\\matsumura\\Documents\\BinSimulator\\XYBin\\bin\\exp\\6DPOSE\\20211119180424"
    shape = "u"
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
    graph = tc6d.make_sim_graph(template, pose, center, cube1_pos)
    num_obj = len(graph)
    writhe_collection, height_collection = tc6d.compute_tangleship(root_dir, graph, pose)

    result_print(f"Writhe: {np.round(writhe_collection, 3)}")
    result_print(f"Height: {np.round(height_collection, 3)}")

    """clustering algorithm"""

    clustering_array = np.concatenate([writhe_collection, height_collection])
    clustering_array = (np.reshape(clustering_array, (2,num_obj))).T

    kmeans = KMeans(n_clusters=2, random_state=0).fit(clustering_array)
    result_print(f"Clustering result: {kmeans.labels_}")

    plt.show()

    # tc6d.find_writhe_thre(writhe_collection, height_collection)

    """visualize all objects and tangleship"""
    fig = plt.figure(figsize=(15, 5))
    ax3d = fig.add_subplot(132, projection='3d')
    ax3d.view_init(167, -87)
    # plot axis
    ax3d.plot([0, 10], [0, 0], [0, 0], color='red')
    ax3d.plot([0, 0], [0, 10], [0, 0], color='green')
    ax3d.plot([0, 1], [0, 0], [0, 10], color='blue')
    ax3d.set_xlim3d(-100, 100)
    ax3d.set_ylim3d(-100, 100)
    ax3d.set_zlim3d(-100, 100)
    ax3d.set_xticklabels([])
    ax3d.set_yticklabels([])
    ax3d.set_zticklabels([])
    plt.title("3D coordinate")
    

    ax1 = fig.add_subplot(131)
    plt.title("Writhe-height relationship")

    sorted_index = list(range(num_obj))
    cmap = get_cmap(len(sorted_index))
    for i, j in zip(sorted_index, range(num_obj)):
        node = graph[i]
        node_cmap = cmap(j)
        ax1.scatter(clustering_array[j][0],clustering_array[j][1], marker='o', color=node_cmap)
        ax1.text(clustering_array[j][0],clustering_array[j][1]+2, str(kmeans.labels_[j]), bbox=dict(boxstyle="round",ec=(1., 0.5, 0.5),fc=(1., 0.8, 0.8),))
        ax = tc6d.draw_node(node, ax3d, alpha=1, color=node_cmap)

    plt.xlabel("writhe")
    plt.ylabel("height")

    ax2 = fig.add_subplot(133)
    im = cv2.imread(im_path)
    rot90 = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ax2.imshow(rot90, cmap='gray')
    plt.title("Depth image")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    main()

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))