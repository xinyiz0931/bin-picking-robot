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
from scipy.spatial.transform import Rotation as R
import itertools
import glob
from utils.plot_utils import plot_subfigures, get_cmap
from utils.base_utils import *
from tangle_solution.topo_coor_6d import TopoCoor6D

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

def process_raw_pc(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    # o3d.visualization.draw_geometries([pcd])

    xyz = np.asarray(pcd.points)
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pc_path = "./vision/tmp/reform.ply"
    o3d.io.write_point_cloud(pc_path, pcd)
    re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)


def process_raw_xyz(xyz_path):
    xyz = np.loadtxt(xyz_path)
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pc_path = "./vision/tmp/reform.ply"
    o3d.io.write_point_cloud(pc_path, pcd)
    re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)

def reform_xyz(xyz):
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pc_path = "./vision/tmp/reform.ply"
    o3d.io.write_point_cloud(pc_path, pcd)
    re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)

def main():
    
    tc6d = TopoCoor6D()

    """Configurations defined by users"""

    # root_dir = "./vision/tmp/tangle_example_1"
    root_dir = "D:\\code\\dataset\\tangle_in_sim\\example_1"
    shape = "s"
    graph = []

    pc_path = os.path.join(root_dir, "point.ply")
    im_path = os.path.join(root_dir, "depth.png")
    pose_path = os.path.join(root_dir, "pose.txt")
    sk_path = f"./objmodel/skeleton_{shape}shape.txt"
    cd_path = f"./objmodel/vhacd_{shape}shape.txt"

    with open(cd_path) as file:
        vhacd = file.readlines()

    center = np.array([float(p) for p in vhacd[0].split(' ')])
    cube1_pos = np.array([float(p) for p in vhacd[3].split(' ')])

    pose = np.loadtxt(pose_path)
    template = np.loadtxt(sk_path, delimiter=',')


    """compute tangleship"""

    graph = tc6d.make_sim_graph(template, pose, center, cube1_pos)
    num_obj = len(graph)
    writhe_collection, height_collection = tc6d.compute_tangleship(root_dir, graph, pose)
    result_print(f"Writhe: {writhe_collection}")
    result_print(f"Height: {height_collection}")

    plt.scatter(writhe_collection, height_collection, marker='o')
    plt.show()

    tc6d.find_writhe_thre(writhe_collection, height_collection)
    """visualize all objects and tangleship"""
    fig = plt.figure(figsize=(15, 6), )
    ax = fig.add_subplot(121, projection='3d')
    ax.view_init(167, -87)

    sorted_index = list(range(num_obj))
    cmap = get_cmap(len(sorted_index))
    for i, j in zip(sorted_index, range(num_obj)):
        node = graph[i]
        node_cmap = cmap(j)
        ax = tc6d.draw_node(node, ax, alpha=1, color=node_cmap)

    ax.plot([0, 10], [0, 0], [0, 0], color='red')
    ax.plot([0, 0], [0, 10], [0, 0], color='green')
    ax.plot([0, 1], [0, 0], [0, 10], color='blue')
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    ax.set_xlim3d(-100, 100)
    ax.set_ylim3d(-100, 100)
    ax.set_zlim3d(-100, 100)

    ax1 = fig.add_subplot(122)
    im = cv2.imread(im_path)
    rot90 = cv2.rotate(im, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ax1.imshow(rot90, cmap='gray')

    # plt.tight_layout()
    # plt.show()


if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    main()

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))