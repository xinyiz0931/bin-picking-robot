import os
import sys
sys.path.append("./")
import cv2
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import itertools
from scipy.spatial.transform import Rotation as R

from utils.transform_utils import *

def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([('', a.dtype)]*a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))

def sample_view():
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    views = []
    phi_xz = 60
    phi_y = 45
    init_view = [0,0,1]
    # stay same as the simulator: first phi_xz, then y
    ax.quiver(0, 0, 0, init_view[0], init_view[1], init_view[2], length = 2, color='black', alpha=0.75)
    for x in np.arange(0,91,phi_xz):
        for y in np.arange(0,360,phi_y):
            if x != 0:
                # print(x,0,z) # euler angle
                rot = np.dot(rpy2mat([x,0, y]), init_view)
                rot =  rot / np.linalg.norm(rot)
                # print([x,y,0], " ===> ", rpy2quat([x,y,0]))
                print([x,y,0], " ===> ", rot)
                ax.quiver(0, 0, 0, rot[0], rot[1], rot[2], length = 2, color='black', alpha=0.25,lw=2)

        # for v in views:
        #     if (q!=v).any():
        #         views.append(q)

    ax.set_xlim3d(-2, 2)
    ax.set_ylim3d(-2, 2)
    ax.set_zlim3d(0, 2)
    ax.set_box_aspect(aspect = (2,2,1))     
    # plt.axis('off')
    ax.plot([0, 0.5], [0, 0], [0, 0], color='blue') # x
    ax.plot([0, 0], [0, 0.5], [0, 0], color='red') # y
    ax.plot([0, 0], [0, 0], [0, 0.5], color='green') # z
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_zticklabels([])

    ax.view_init(27,28)
    plt.show()

def test_view():
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

    p = np.array([[20,30,40],[20,40,0]])
    p2 = np.array([[-20,0,-40],[-20,-10,0]])
    from tangle_solution.topo_coor_6d import TopoCoor6D

    tp = TopoCoor6D()
    tp.draw_node(p, ax3d, "black")
    tp.draw_node(p2, ax3d, "purple")
    ax2 = fig.add_subplot(122)
    ax2.plot([0, 10], [0, 0], color='red') # x
    ax2.plot([0, 0], [0, 10], color='blue') # z
    plt.xlim(-120, 120)
    plt.ylim(120, -120)

    # tp.draw_projection_node_2d(p, [0,0,0], ax2, 'black', alpha=0.5)
    tp.draw_projection_node_2d(p, [180,-90,0], ax2, 'black', alpha=1)

    tp.draw_projection_node_2d(p2, [180,-90,0], ax2, 'purple', alpha=1)

    plt.show()

def show_mesh_model():
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh("./objmodel/model_u.ply")
    print("Testing mesh in Open3D...")
    print(mesh)
    print('Vertices:')
    print(np.asarray(mesh.vertices))
    print('Triangles:')
    print(np.asarray(mesh.triangles))
    print("Computing normal and rendering it.")
    mesh.compute_vertex_normals()
    print(np.asarray(mesh.triangle_normals))
    o3d.visualization.draw_geometries([mesh])

def create_img():
    img = cv2.imread("C:\\Users\\matsumura\\Documents\\BinSimulator\\XYBin\\bin\\exp\\6DPOSE\\20211212161104\\grasp.png")
    drawn = cv2.circle(img, (159,272), 7, (0,192,255), -1)
    cv2.imwrite("C:\\Users\\matsumura\\Documents\\BinSimulator\\XYBin\\bin\\exp\\6DPOSE\\20211212161104\\result.png", drawn)
    plt.imshow(img)
    plt.show()
def main():
    create_img()
    # img = cv2.imread("./vision/depth/depth0.png",0)
    # _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)
    # plt.imshow(img, cmap='gray')
    # plt.imshow(mask*0.5, cmap='jet', alpha=0.3)
    # plt.show()
if __name__ == '__main__':
    main()