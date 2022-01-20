import os
import sys
import glob
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import math
import random
import numpy as np
import itertools

np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from scipy import ndimage
from utils.base_utils import *
from utils.vision_utils import *
from utils.plot_utils import *
from utils.transform_utils import *
            
class TopoCoor6D(object):
    def __init__(self, this=0):
        self.this = this

    def cross_product(self, a, b):
        x1, y1, z1 = a
        x2, y2, z2 = b
        return np.array([(y1 * z2 - y2 * z1), -(x1 * z2 - x2 * z1), (x1 * y2 - x2 * y1)])

    def gli_original(self, line1, line2):
        """
        Calculate Gaussian link integral in a geometrical way
        Input: two line segment line = [p1.x,p1.y.p1.z, p2.x,p2.y.p2.z]
        output: w [float]
        """
        a = np.array([line1[0], line1[1], line1[2]])
        b = np.array([line1[3], line1[4], line1[5]])

        c = np.array([line2[0], line2[1], line2[2]])
        d = np.array([line2[3], line2[4], line2[5]])

        r_ab = b - a
        r_cd = d - c
        r_ac = c - a
        r_ad = d - a
        r_bc = c - b
        r_bd = d - b
        

        c_ac_ad = self.cross_product(r_ac, r_ad)
        c_ad_bd = self.cross_product(r_ad, r_bd)
        c_bd_bc = self.cross_product(r_bd, r_bc)
        c_bc_ac = self.cross_product(r_bc, r_ac)

        n_a = c_ac_ad / np.linalg.norm(c_ac_ad)
        n_b = c_ad_bd / np.linalg.norm(c_ad_bd)
        n_c = c_bd_bc / np.linalg.norm(c_bd_bc)
        n_d = c_bc_ac / np.linalg.norm(c_bc_ac)

        w = (np.arcsin(np.dot(n_a, n_b)) + np.arcsin(np.dot(n_b, n_c)) + np.arcsin(np.dot(n_c, n_d)) + np.arcsin(
            np.dot(n_d, n_a)))/(4*math.pi)

        return np.nan_to_num(w)

    def gli_2d(self, line1, line2):
        """
        Calculate Gaussian link integral in a geometrical way
        Input: two line segment line = [p1.x,p1.y, p2.x,p2.y]
        output: w [float]
        """
        a = np.array([line1[0], line1[1], 0])
        b = np.array([line1[2], line1[3], 0])

        c = np.array([line2[0], line2[1], 0])
        d = np.array([line2[2], line2[3], 0])

        r_ab = b - a
        r_cd = d - c
        r_ac = c - a
        r_ad = d - a
        r_bc = c - b
        r_bd = d - b
        

        c_ac_ad = self.cross_product(r_ac, r_ad)
        c_ad_bd = self.cross_product(r_ad, r_bd)
        c_bd_bc = self.cross_product(r_bd, r_bc)
        c_bc_ac = self.cross_product(r_bc, r_ac)

        n_a = c_ac_ad / np.linalg.norm(c_ac_ad)
        n_b = c_ad_bd / np.linalg.norm(c_ad_bd)
        n_c = c_bd_bc / np.linalg.norm(c_bd_bc)
        n_d = c_bc_ac / np.linalg.norm(c_bc_ac)

        w = (np.arcsin(np.dot(n_a, n_b)) + np.arcsin(np.dot(n_b, n_c)) + np.arcsin(np.dot(n_c, n_d)) + np.arcsin(
            np.dot(n_d, n_a)))/(4*math.pi)

        return np.nan_to_num(w)
    
    def gli_sign(self, line1, line2):
        """
        Calculate the sign of Gaussian link integral in a geometrical way
        Check if the tanlge is right-handed or not

        Input: two line segment line = [p1.x,p1.y.p1.z, p2.x,p2.y.p2.z]
        output: s {int} - +1 or -1
        """
        a = np.array([line1[0], line1[1], line1[2]])
        b = np.array([line1[3], line1[4], line1[5]])

        c = np.array([line2[0], line2[1], line2[2]])
        d = np.array([line2[3], line2[4], line2[5]])

        r_ab = b - a
        r_cd = d - c
        r_ac = c - a
        return np.sign(np.dot(self.cross_product(r_cd, r_ab), r_ac))
    
    def node2edge(self, node):
        """
        :param graph: positions of each node, N: node number
                    graph = N x 3
        :return: positions of each edge, N: node number
                    return: (N-1) x 6
        """
        num = node.shape[0]
        edge_collection = np.array([])
        for i in range(num - 1):
            edge_collection = np.append(edge_collection, [node[i], node[i + 1]])
        return edge_collection.reshape(num - 1, 6)

    def draw_node(self, node, draw_ax, color, alpha=1):
        # color = tuple([x / 255 for x in color])
        draw_ax.scatter(node[1:, 0], node[1:, 1], node[1:, 2], color=color, alpha=alpha)
        for i in range(node.shape[0] - 1):
            draw_ax.plot([node[i][0], node[i + 1][0]], [node[i][1], node[i + 1][1]], [node[i][2], node[i + 1][2]],
                        color=color, alpha=alpha)
        # additional: drawing the start(red)/end(yellow) node
        # draw_ax.scatter(node[0][0], node[0][1], node[0][2], color='red', alpha=alpha)
        # draw_ax.scatter(node[-1][0], node[-1][1], node[-1][2], color='yellow', alpha=alpha)
        return draw_ax

    def draw_projection_node_2d(self, node, euler_angle, draw_ax, color, alpha=1):
        """
        1. rotate along x-axis for some degrees
        2. project on a 2d planar
        Arguments:
            node {array} - shape=(N,3)
            eular_angle {list} - [r,p,y]
        """
        projection_mat = simrpy2mat(euler_angle)
        node_proj = (np.dot(projection_mat, node.T)).T
        node_proj[:,1]=0
        draw_ax.scatter(node_proj[1:, 0], node_proj[1:, 2], color=color, alpha=alpha)
        for i in range(node_proj.shape[0] - 1):
            draw_ax.plot([node_proj[i][0], node_proj[i + 1][0]], [node_proj[i][2], node_proj[i + 1][2]],
                        color=color, alpha=alpha)
        # additional: drawing the start(red)/end(yellow) node
        # draw_ax.scatter(node_proj[0][0], node_proj[0][2], color='red', alpha=alpha)
        # draw_ax.scatter(node_proj[-1][0], node_proj[-1][2], color='yellow', alpha=alpha)
        return draw_ax

    def draw_projection_node_3d(self, node, euler_angle, draw_ax, color, alpha=1):
        """
        1. rotate along x-axis for some degrees
        2. project on a 2d planar
        Arguments:
            node {array} - shape=(N,3)
            eular_angle {list} - [r,p,y]
        """
        
        # projection_mat = rpy2mat(euler_angle)
        projection_mat = simrpy2mat(euler_angle)
        node_proj = (np.dot(projection_mat, node.T)).T
        # node_proj = rotate_in_sim(node, xz, y)
        draw_ax.scatter(node_proj[1:, 0], node_proj[1:, 1], node_proj[1:, 2], color=color, alpha=alpha)
        for i in range(node_proj.shape[0] - 1):
            draw_ax.plot([node_proj[i][0], node_proj[i + 1][0]], [node_proj[i][1], node_proj[i + 1][1]], [node_proj[i][2], node_proj[i + 1][2]],
                        color=color, alpha=alpha)
        # additional: drawing the first node
        draw_ax.scatter(node_proj[0][0], node_proj[0][1], node_proj[0][2], color='red', alpha=alpha)
        draw_ax.scatter(node_proj[-1][0], node_proj[-1][1],node_proj[-1][2], color='yellow', alpha=alpha)
        return draw_ax

    def draw_projection_node_new(self, node, euler_angle, draw_ax, color, alpha=0.4):
        # 1. projection on x-z plane
        # node[:,1]=0
        # 2. rotate along x-axis for some degrees
        projection_mat = rpy2mat(euler_angle)
        # 3. start drawing
        node_proj = (np.dot(projection_mat, node.T)).T
        node_proj[:,1]=0
        draw_ax.scatter(node_proj[1:, 0], node_proj[1:, 1], node_proj[1:, 2], color=color, alpha=alpha)
        for i in range(node_proj.shape[0] - 1):
            draw_ax.plot([node_proj[i][0], node_proj[i + 1][0]], [node_proj[i][1], node_proj[i + 1][1]], [node_proj[i][2], node_proj[i + 1][2]],
                        color=color, alpha=alpha)
        # additional: drawing the first node
        draw_ax.scatter(node_proj[0][0], node_proj[0][1], node_proj[0][2], color='red', alpha=alpha)
        # draw_ax.plot([0,0], [100,0],[0,0], alpha=0.3, color='yellow', marker='^')
        return draw_ax

    def compute_writhe(self, graph):
        """compute writhe matrix between multiple same objects

        Arguments:
            graph {array} -- shape=(N x M x 3)
                          -- N: number of objs
                          -- M: number of template nodes
        Returns:
            obj_wmat [array] -- matrix of all object writhe
        """
        # segments of both graphs
        graph = np.array(graph)
        n_obj = graph.shape[0]
        objs = []
        obj_wmat = np.zeros((n_obj, n_obj))

        for nodes in graph:
            # obj number
            objs.append(self.node2edge(nodes))
        for i in range(n_obj):
            obj1 = objs[i]
            for j in range(i + 1, n_obj):
                gli_sum = 0
                # obj #i & obj #j
                obj2 = objs[j]
                for (seg1, seg2) in list(itertools.product(obj1, obj2)):
                    gli_sum += gli_original(seg1, seg2)
                obj_wmat[i][j] = gli_sum
                obj_wmat[j][i] = gli_sum
        # gli_mat = gli_mat + gli_mat.T
        return obj_wmat.sum(axis=1)

    def compute_avg_height(self, graph):
        """compute height among multiple same objects
        Arguments:
            graph {array} -- shape=(N x M x 3)
                          -- N: number of objs
                          -- M: number of template nodes
        """
        # segments of both graphs
        graph = np.array(graph)
        n_obj = graph.shape[0]
        avhglist_collection = []
        for nodes in graph:
            avhglist_collection.append(np.mean(nodes[:1]))
        return avhglist_collection
    
    def get_obj_height(self, nodes):
        """compute height for one object represented by node
        Arguments:
            node {array} -- shape=(M x 3)
                          -- M: number of template nodes
        """
        return np.mean(nodes[:1])

    def get_proj_height(self, nodes, proj_angle):
        new_nodes = (np.dot(simrpy2mat(proj_angle), nodes.T)).T
        return np.mean(new_nodes[:1])

    def compute_writhe_matrix(self, node1, node2):
        """compute writhe matrix between two arbitrary objects

        Arguments:
            node1 {array} -- shape=(num1, 3)
            node2 {array} -- shape=(num2, 3)

        Returns:
            wmat [array] -- writhe matrix
        """
        edge1 = self.node2edge(node1)
        edge2 = self.node2edge(node2)

        n1_seg = edge1.shape[0]
        n2_seg = edge2.shape[0]
        wmat = np.zeros((n1_seg, n2_seg))
        for i in range(n1_seg):
            for j in range(n2_seg):
                gli = self.gli_original(edge1[i], edge2[j])
                wmat[i][j] = gli
        return wmat

    def compute_writhe_matrix(self, graph, i, j):
        """graph: shape=(2(only 2 objs) x M(template nodes) x 3)"""
        graph = np.array(graph)
        objs = []
        for nodes in graph:
            # obj number
            objs.append(self.node2edge(nodes))

        objs = np.array(objs)

        n_seg = objs.shape[1]

        gli_mat = np.zeros((n_seg, n_seg))
        obj1 = objs[i]
        obj2 = objs[j]
        for k in range(n_seg):
            for t in range(n_seg):
                gli = self.gli_original(obj1[k], obj2[t])
                gli_mat[k][t] = gli
        return gli_mat

    # def transfer_sim_pos(self, node, p, q, center, cube1_pos):
    #     trans = center - cube1_pos
    #     m = quat2mat(q)
    #     trans_ = np.array([np.dot(trans, m[0]), np.dot(trans, m[1]), np.dot(trans, m[2])])
    #     p += trans_
    #     return rotate_3d(node, m) + p
    def transfer_sim_pos(self, node, p, q, cube1_pos):
        trans = -cube1_pos
        m = quat2mat(q)
        trans_ = np.array([np.dot(trans, m[0]), np.dot(trans, m[1]), np.dot(trans, m[2])])
        p += trans_
        return rotate_3d(node, m) + p


    def make_sim_graph(self, template, pose, cube1_pos):
        graph = []
        for v in pose:
            # make a new template
            node = template.copy()
            pos = v[0:3]
            qua = v[3:7]
            # translation P, rotation R
            rot = quat2mat(qua)
            P = pos - cube1_pos
            rot_ori = pos
            node = self.transfer_sim_pos(node, pos, qua, cube1_pos)
            # node = rotate_3d(node + P, rot, origin=(rot_ori[0], rot_ori[1], rot_ori[2]))
            graph.append(node)
        return graph

    def check_overlap(self, mask_dir, i, j):
        """
        Return:
            not overlap / singulated -> false
            overlap -> true
        """
        query_mask_i = os.path.join(mask_dir, f"mask_{i}.png")
        query_mask_j = os.path.join(mask_dir, f"mask_{j}.png")
        mask_i = cv2.imread(query_mask_i, 0)
        mask_j = cv2.imread(query_mask_j, 0)

        # set to binary
        mask_i[mask_i == 25] = 0
        mask_i[mask_i > 26] = 1

        mask_j[mask_j == 25] = 0
        mask_j[mask_j > 26] = 1

        return bool(np.count_nonzero(mask_i & mask_j))
        # return (mask_i & mask_j == 0).all()

    def compute_tangleship_old(self, root_dir, graph, pose):
        """compute height among multiple same objects
        Arguments:
            graph {array} -- shape=(N x M x 3)
                            -- N: number of objs
                            -- M: number of template nodes
            pose {array}  -- shape=(N x 6)
        """
        # segments of both graphs
        graph = np.array(graph)
        n_obj = graph.shape[0]
        n_seg = graph.shape[1]-1
        main_proc_print("Object number is {} and segment number is {}".format(n_obj, n_seg))
        objs = []

        gli_mat = np.zeros((n_obj, n_obj))
        for nodes in graph:
            # obj number
            objs.append(self.node2edge(nodes))
        objs = np.array(objs)

        # get their heights
        avhglist = self.compute_avg_height(graph)

        # get their writhe & twist
        for i in range(n_obj):
            obj1 = objs[i]
            for j in range(i + 1, n_obj):
                gli_sum = 0
                # obj #i & obj #j
                obj2 = objs[j]
                for (seg1, seg2) in list(itertools.product(obj1, obj2)):
                    gli_sum += self.gli_original(seg1, seg2)
                # gli_sum = gli_sum / n_seg
                gli_sum = gli_sum * 2
                if gli_sum >= 2:
                    result_print(f"Obj {i} & obj {j}: {gli_sum}")
                gli_mat[i][j] = gli_sum
                gli_mat[j][i] = gli_sum
        return gli_mat
        # glilist = (gli_mat + gli_mat.T).sum(axis=1)
        
        # return glilist, avhglist

    def compute_tangleship(self, root_dir, graph, pose):
        """compute height among multiple same objects
        Arguments:
            graph {array} -- shape=(N x M x 3)
                            -- N: number of objs
                            -- M: number of template nodes
            pose {array}  -- shape=(N x 6)
        """
        # segments of both graphs
        graph = np.array(graph)
        n_obj = graph.shape[0]
        n_other_obj = graph.shape[0] - 1
        n_seg = graph.shape[1]-1
        main_proc_print("Object number is {} and segment number is {}".format(n_obj, n_seg))
        objs = []
        objs_proj = []
        voting = [0] * n_obj

        gli_mat = np.zeros((n_obj, n_obj))

        for nodes in graph:
            objs.append(self.node2edge(nodes))
            # nodes[:,1] = 0
            # projection using euler angle
            projection_mat = rpy2mat([0,0,45])
            # nodes = (np.dot(projection_mat, nodes.T)).T
            nodes[:,1]=0
            objs_proj.append(self.node2edge(nodes)) # projection on 2d planar

        objs = np.array(objs)
        objs_proj = np.array(objs_proj)
        # get their writhe
        for i in range(n_obj):
            obj1 = objs_proj[i]
            other_obj_proj = np.delete(objs_proj, i, axis=0)
            other_obj = np.delete(objs, i ,axis=0)
            gli_sum =0
            for j in range(other_obj_proj.shape[0]):
                obj2 = other_obj_proj[j]
                for (k, t) in list(itertools.product(range(n_seg), range(n_seg))):
                    seg1_proj, seg2_proj = obj1[k], obj2[t]
                    seg1, seg2 = objs[i][k], other_obj[j][t]

                    gli = self.gli_original(seg1_proj, seg2_proj)
                    gli_sum += gli

                    gli_sign = self.gli_sign(seg1, seg2)
                    if gli*2>=1:
                        warning_print(f"Obj {i} crossing attention! gli={gli}, sign={gli_sign}")
                        # print(calc_intersection((seg1[0], seg1[2]), (seg1[3], seg1[5]), (seg2[0], seg2[2]), (seg2[3], seg2[5])))
                        # if objs[i][k][1] > objs[j][t][1]:
                        #     voting[j] += 1
            gli_sum = gli_sum * 2
            result_print(f"Obj {i} has {gli_sum} crossings with others! ")
            if gli_sum ==0:
                important_print(f"Obj {i} can be picked! ")
                voting[i] = 1
        result_print(voting)
        return voting

    def compute_tangleship_with_projection(self, root_dir, graph, pose, proj_angle=[0,0,0]):
        """compute height among multiple same objects
        Arguments:
            graph {array} -- shape=(N x M x 3)
                            -- N: number of objs
                            -- M: number of template nodes
            pose {array}  -- shape=(N x 6)
            proj_angle {list} -- [roll, pitch, yaw] with degrees
        """

        graph = np.array(graph)

        n_obj = graph.shape[0]
        n_other_obj = graph.shape[0] - 1
        n_seg = graph.shape[1]-1
        main_proc_print("Object number is {} and segment number is {}".format(n_obj, n_seg))
        rot_graph = []
        objs = []
        objs_proj = []
        voting = [0] * n_obj

        labels = {} 

        gli_mat = np.zeros((n_obj, n_obj))

        # check if projection neede
        if proj_angle != [0,0,0]:
            main_proc_print(f"Start projecting along {proj_angle} ...")
        
        # rotate and normalize the object'
        plane_minh = 9999
        for nodes in graph:
            # projecting nodes using euler angle
            new_nodes = (np.dot(simrpy2mat(proj_angle), nodes.T)).T
            rot_graph.append(new_nodes)
            if np.min(new_nodes[:,1]) < plane_minh: 
                plane_minh = np.min(new_nodes[:,1])
        rot_graph = np.array(rot_graph)
        if plane_minh < 0:
            rot_graph += np.array([0, -plane_minh, 0])

        # record the original and projected locations
        for nodes in rot_graph:
            objs.append(self.node2edge(nodes)) # 3d
            nodes[:,1]=0
            objs_proj.append(self.node2edge(nodes)) # 2d

        objs = np.array(objs)
        # get their writhe (number of crossings)
        for i in range(n_obj):
            # calculate writhe between objs[i] nad other_obj[j]
            # other_obj_proj = np.delete(objs_proj, i, axis=0)
            # other_obj = np.delete(objs, i ,axis=0)
            
            writhe =0
            labels[i] = []
            cross_num= 0

            for j in range(n_obj):
                for (k, t) in list(itertools.product(range(n_seg), range(n_seg))):
                    seg1_proj, seg2_proj = objs_proj[i][k], objs_proj[j][t]
                    seg1, seg2 = objs[i][k], objs[j][t]

                    gli = self.gli_original(seg1_proj, seg2_proj)
                    if gli*2 == 1:
                        crossing = calc_intersection([seg1[0], seg1[2]], [seg1[3], seg1[5]], 
                                                [seg2[0], seg2[2]], [seg2[3], seg2[5]])
                        # label the crossings for obj i - seg1, seg2, crossing 
                        d1 = calc_lineseg_dist([crossing[0],0,crossing[1]], seg1)
                        d2 = calc_lineseg_dist([crossing[0],0,crossing[1]], seg2)

                        if d1 > d2:
                            labels[i].append(1)
                        elif d1 < d2:
                            labels[i].append(-1)
                        cross_num += 1
                    writhe += gli
            writhe *= 2

        print(labels)
        return labels

if __name__ == "__main__":
    tp6d = TopoCoor6D()
    fig = plt.figure(figsize=(15, 7))
    ax3d = fig.add_subplot(111, projection='3d')
    seg1 = [-18.70203845,  36.16549733,  -7.66789376 , -9.19257037,  38.7589582, -10.83349299]
    seg2 = [ -8.03589864 , 22.51008021, -28.75285483, -15.13018445,  69.99675389, -6.50976386]

    seg1 = [-10.56096588,  72.94437274, -16.26506441, -59.32415402,  90.29244468,-5.25306886] 
    seg2 = [ -8.03589864,  22.51008021, -28.75285483, -15.13018445,  69.99675389, -6.50976386]

    seg1 = [-10.56096588,  72.94437274, -16.26506441, -59.32415402,  90.29244468, -5.25306886] 
    seg2 = [-40.03033631,  15.41193082,  -5.81728425, -38.06116242,   6.40655368, -10.53105332]


    seg1 = [ 18.70203845,  90.90582275,  -7.66789376,   9.19257037 , 88.31236188, -10.83349299] 
    seg2= [  8.03589864, 104.56123987, -28.75285483 , 15.13018445 , 57.0745662, -6.50976386]
    seg1 = [ 10.56096588,  54.12694735, -16.26506441,  59.32415402 , 36.77887541, -5.25306886] 
    seg2 = [ 40.03033631, 111.65938926,  -5.81728425 , 38.06116242, 120.6647664, -10.53105332]
    seg1= [ 10.56096588,  54.12694735, -16.26506441 , 59.32415402,  36.77887541, -5.25306886]
    seg2 = [  8.03589864, 104.56123987, -28.75285483 , 15.13018445,  57.0745662, -6.50976386]

    crossing = calc_intersection([seg1[0], seg1[2]], [seg1[3], seg1[5]], 
                                                [seg2[0], seg2[2]], [seg2[3], seg2[5]])
    print(crossing)

    # plt.plot([seg1[0], seg1[3]], [seg1[2], seg1[5]],color='black', alpha=0.3)
    # plt.plot([seg2[0], seg2[3]], [seg2[2], seg2[5]],color='black', alpha=0.3)
    # plt.scatter(crossing[0],crossing[1],color='orange', alpha=0.3)
    # plt.show()

    d1 = calc_lineseg_dist([crossing[0],0,crossing[1]], seg1)
    d2 = calc_lineseg_dist([crossing[0],0,crossing[1]], seg2)
    print(d1, d2)



    # l1_proj = [-10.56096588, 0, -16.26506441, -59.32415402, 0, -5.25306886] 
    # l2_proj = [-40.03033631,  0,  -5.81728425, -38.06116242, 0, -10.53105332]

    # gli = tp6d.gli_original(l1_proj, l2_proj)
    # print("gli = ", gli)
    
    ax3d.plot([seg1[0], seg1[3]], [seg1[1],seg1[4]], [seg1[2], seg1[5]],color='blue')
    ax3d.plot([seg2[0], seg2[3]], [seg2[1],seg2[4]], [seg2[2], seg2[5]],color='red')

    ax3d.plot([seg1[0], seg1[3]], [0,0], [seg1[2], seg1[5]],color='black', alpha=0.3)
    ax3d.plot([seg2[0], seg2[3]], [0,0], [seg2[2], seg2[5]],color='black', alpha=0.3)

    ax3d.scatter(crossing[0],0,crossing[1],color='orange', alpha=0.3)
    ax3d.scatter(crossing[0],d1,crossing[1],color='blue')
    ax3d.scatter(crossing[0],d2,crossing[1],color='red')
    plt.show()

    
    # ax3d.plot([l[0], l[3]], [0,0], [l[2], l[5]],color='black', alpha=0.3)
    # ax3d.scatter(cp[0],0,cp[1],color='orange', alpha=0.3)
    # ax3d.scatter(cp[0],d1,cp[1],color='orange')
    # ax3d.scatter(cp[0],d2,cp[1],color='orange')
    # plt.show()

    # plt.plot([l1[0], l1[3]], [l1[2], l2[5]],color='black', alpha=0.3)
    # plt.plot([l2[0], l2[3]], [l2[2], l2[5]],color='black', alpha=0.3)
    # plt.scatter(cp[0],cp[1],color='orange', alpha=0.3)
    # plt.show()
    
    # lines:  [-18.70203845  36.16549733  -7.66789376  -9.19257037  38.7589582
#  -10.83349299] [ -8.03589864  22.51008021 -28.75285483 -15.13018445  69.99675389
#   -6.50976386]
# Obj 0:seg 2 ==> [-14.292649476783435,36.46286407364223,-9.135731731780636] | Obj 1:seg 7 ==>  [-14.292649476783435,30.50708378104471,-9.135731731780636]
# lines:  [-10.56096588  72.94437274 -16.26506441 -59.32415402  90.29244468
#   -5.25306886] [-40.03033631  15.41193082  -5.81728425 -38.06116242   6.40655368
#  -10.53105332]
# Obj 0:seg 7 ==> [-38.28083876041251,78.22951260369001,-10.005196497364514] | Obj 1:seg 1 ==>  [-38.28083876041251,6.431851456870234,-10.005196497364514]
# lines:  [-10.56096588  72.94437274 -16.26506441 -59.32415402  90.29244468
#   -5.25306886] [ -8.03589864  22.51008021 -28.75285483 -15.13018445  69.99675389
#   -6.50976386]
# Obj 0:seg 7 ==> [-12.131947309442216,72.96215016762046,-15.910295961991459] | Obj 1:seg 7 ==>  [-12.131947309442216,26.237618872345625,-15.910295961991459]


# # l1 = [1,0,0,1,3,0]
# # l2 = [0,1,0,3,1,0]

# # tp6d = TopoCoor6D()
# # writhe = tp6d.gli_original(l1, l2)
# # print(writhe)
#     tp6d = TopoCoor6D()

#     l1 = [1,0,1,1,2,1]
#     # l2 = [1,0,1,0,2,0]
#     # l2 = [0,1,0.5,2,1,0.5]
#     l2 = [0,1,0.2,2,1,0.5] # same plane

#     l1_proj = [1,0,-0.5,1,2,-0.5]
#     # l2 = [1,0,1,0,2,0]
#     # l2 = [0,1,0.5,2,1,0.5]
#     l2_proj = [0,1,-0.5,2,1,-0.5] # same plane

#     cp = calc_intersection([l1_proj[0], l1_proj[1]], [l1_proj[3], l1_proj[4]],
#                           [l2_proj[0], l2_proj[1]], [l2_proj[3], l2_proj[4]])
#     print(cp)

    

#     d1 = calc_lineseg_dist([cp[0],cp[1], 0], l1)
#     print("Found it! ", d1)
#     d2 = calc_lineseg_dist([cp[0],cp[1], 0], l2)
#     print("Found it! ", d2)

    

#     # s = np.vstack([a1,a2,b1,b2])        # s for stacked
#     # h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
#     # l1 = np.cross(h[0], h[1])           # get first line
#     # l2 = np.cross(h[2], h[3])           # get second line

#     lc1 = np.cross(l1[0:3], l1[3:6])
#     lc2 = np.cross(l2[0:3], l2[3:6])
#     print("line function: ", lc1, lc2)

#     a = np.array([l1[0], l1[1], l1[2]])
#     b = np.array([l1[3], l1[4], l1[5]])
#     c = np.array([l2[0], l2[1], l2[2]])
#     d = np.array([l2[3], l2[4], l2[5]])

#     r_ab = b - a
#     r_cd = d - c
#     r_ac = c - a
#     r_ad = d - a
#     r_bc = c - b
#     r_bd = d - b
    

#     c_ac_ad = tp6d.cross_product(r_ac, r_ad)
#     c_ad_bd = tp6d.cross_product(r_ad, r_bd)
#     c_bd_bc = tp6d.cross_product(r_bd, r_bc)
#     c_bc_ac = tp6d.cross_product(r_bc, r_ac)

#     n_a = c_ac_ad / np.linalg.norm(c_ac_ad)
#     n_b = c_ad_bd / np.linalg.norm(c_ad_bd)
#     n_c = c_bd_bc / np.linalg.norm(c_bd_bc)
#     n_d = c_bc_ac / np.linalg.norm(c_bc_ac)

#     normal = tp6d.cross_product(r_cd, r_ab)
#     normal = normal / np.linalg.norm(normal)

#     print(normal)

#     w = (np.arcsin(np.dot(n_a, n_b)) + np.arcsin(np.dot(n_b, n_c)) + np.arcsin(np.dot(n_c, n_d)) + np.arcsin(
#         np.dot(n_d, n_a)))/(4*math.pi)
#     w = np.nan_to_num(w)

#     n = [0,0,-1]

    
#     print(w*2)

#     fig = plt.figure()
#     ax3d = fig.add_subplot(111, projection='3d')
#     ax3d.set_xticklabels([])
#     ax3d.set_yticklabels([])
#     ax3d.set_zticklabels([])

#     ax3d.set_zlim3d(-1, 1)

#     cmap = get_cmap(2)

#     # plot start and end point
#     ax3d.scatter(a[0],a[1],a[2],color='red')
#     ax3d.scatter(c[0],c[1],c[2],color='red')

#     ax3d.plot([l1[0],l1[3]], [l1[1],l1[4]], [l1[2],l1[5]], color=cmap(0))
#     ax3d.plot([l2[0],l2[3]], [l2[1],l2[4]], [l2[2],l2[5]], color=cmap(1)) 

#     ax3d.plot([l1_proj[0],l1_proj[3]], [l1_proj[1],l1_proj[4]], [l1_proj[2],l1_proj[5]], color=cmap(0), alpha=0.3)
#     ax3d.plot([l2_proj[0],l2_proj[3]], [l2_proj[1],l2_proj[4]], [l2_proj[2],l2_proj[5]], color=cmap(1), alpha=0.3) 

#     # plot view direction
#     # ax3d.scatter(0,0,0,color='red')
#     # ax3d.plot([0, n[0]], [0, n[1]], [0, n[2]], color='red', alpha=0.3)


#     # c_ac_ad = tp6d.cross_product(r_ac, r_ad)
#     # c_ad_bd = tp6d.cross_product(r_ad, r_bd)
#     # c_bd_bc = tp6d.cross_product(r_bd, r_bc)
#     # c_bc_ac = tp6d.cross_product(r_bc, r_ac)

#     # # plot n_a, ..., n_d
#     # ax3d.plot([a[0], n_a[0]+a[0]], [a[1], n_a[1]+a[1]], [a[2], n_a[2]+a[2]], color='purple', alpha=0.3)
#     # ax3d.plot([d[0], n_b[0]+d[0]], [d[1], n_b[1]+d[1]], [d[2], n_b[2]+d[2]], color='purple', alpha=0.3)
#     # ax3d.plot([b[0], n_c[0]+b[0]], [b[1], n_c[1]+b[1]], [b[2], n_c[2]+b[2]], color='purple', alpha=0.3)
#     # ax3d.plot([c[0], n_d[0]+c[0]], [c[1], n_d[1]+c[1]], [c[2], n_d[2]+c[2]], color='purple', alpha=0.3)

#     # ax3d.plot([a[0], c[0]], [a[1], c[1]], [a[2], c[2]], color='green', alpha=0.3)
#     # ax3d.plot([b[0], c[0]], [b[1], c[1]], [b[2], c[2]], color='green', alpha=0.3)
#     # ax3d.plot([a[0], d[0]], [a[1], d[1]], [a[2], d[2]], color='green', alpha=0.3)
#     # ax3d.plot([b[0], d[0]], [b[1], d[1]], [b[2], d[2]], color='green', alpha=0.3)

#     # ax3d.plot([a[0], normal[0]+a[0]], [a[1], normal[1]+a[1]], [a[2], normal[2]+a[2]], color='red', alpha=0.3)
#     # ax3d.plot([c[0], normal[0]+c[0]], [c[1], normal[1]+c[1]], [c[2], normal[2]+c[2]], color='red', alpha=0.3)
#     # plt.plot([l1[0],l1[3]], [l1[1],l1[4]])
#     # plt.plot([l2[0],l2[3]], [l2[1],l2[4]], color='orange')

#     ax3d.scatter(cp[0], cp[1], -0.5, color='orange')
    
#     # plt.axis('off')
#     # ax3d.set_box_aspect(aspect = (1,1,1))
#     plt.show()

