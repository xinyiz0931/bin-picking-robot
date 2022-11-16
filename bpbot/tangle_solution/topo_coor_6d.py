import os
import sys
import cv2
import math
import random
import numpy as np
import itertools
import matplotlib.pyplot as plt
from bpbot.utils import *
from bpbot.tangle_solution import TangleObjSke

class TopoCoor6D(object):
    def __init__(self, skeleton_file, collision_file):
        with open(collision_file) as cf:
            cubes = cf.readlines()
        self.center = np.array([float(p) for p in cubes[0].split(' ')])
        self.cube1 = np.array([float(p) for p in cubes[3].split(' ')])

        tok = TangleObjSke()
        self.template = np.array((tok.load_obj_from_json(skeleton_file))["node"])

    def cross_product(self, a, b):
        x1, y1, z1 = a
        x2, y2, z2 = b
        return np.array([(y1 * z2 - y2 * z1), -(x1 * z2 - x2 * z1), (x1 * y2 - x2 * y1)])

    def gli(self, line1, line2):
        """
        Calculate Gaussian link integral in a geometrical way
        Parameters: 
            line1 {list} -- (p1.x,p1.y,p1.z, p2.x,p2.y,p2.z)
            line2 {list} -- (p1.x,p1.y,p1.z, p2.x,p2.y,p2.z)
        Returns: 
            w {float} -- writhe, sun of gli for all segment pairs
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
        Calculate Gaussian link integral in a geometrical way for 2d lines
        Parameters: 
            line1 {list} -- (p1.x,p1.y, p2.x,p2.y)
            line2 {list} -- (p1.x,p1.y, p2.x,p2.y)
        Returns: 
            w {float} -- writhe, sun of gli for all segment pairs
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
        Check if the crossing is right-handed or not for two line segment
        +1 -> right-handed, -1 -> left-handed
        Parameters: 
            line1 {list} -- (p1.x,p1.y,p1.z, p2.x,p2.y,p2.z)
            line2 {list} -- (p1.x,p1.y,p1.z, p2.x,p2.y,p2.z)
        Returns: 
            s {int} - +1 or -1
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
        Parameters:
            node {array} -- (node num x 3), position of each node
        Return: 
            {array} -- (node num-1 x 6), position of each edge
        """
        num = node.shape[0]
        edge_collection = np.array([])
        for i in range(num - 1):
            edge_collection = np.append(edge_collection, [node[i], node[i + 1]])
        return edge_collection.reshape(num - 1, 6)

    def draw_node_3d(self, nodes, ax, color, alpha=1):
        """
        Draw one object in 3D space
        Parameters:
            nodes {array} -- (node num x 3), node of one object
            ax {AxesSubplot} -- matplotlib subplot
            color {tuple} -- (r,g,b,a) normlized
            alpha {float}
        Returns:
            ax {AxesSubplot} -- drawn subplot
        """
        for i in range(nodes.shape[0] - 1):
            ax.plot([nodes[i][0], nodes[i + 1][0]], [nodes[i][1], nodes[i + 1][1]], [nodes[i][2], nodes[i + 1][2]],
                        color=color, alpha=alpha)
        return ax

    def draw_node_2d(self, nodes, ax, color, alpha=1, proj_ea=None):
        """
        Draw one object in 2D space
        1) rotate along x-axis for some degrees 2) project
        Parameters:
            nodes {array} -- (node num x 3), node of one object
            ax {AxesSubplot} -- matplotlib subplot
            color {tuple} -- (r,g,b,a) normlized
            alpha {float}
            proj_ea {list} -- [y,-xz,0], project eular angle (yzx seq using sampled views)
        Returns:
            ax {AxesSubplot} -- drawn subplot
        """
        if proj_ea is None:
            for i in range(nodes.shape[0] - 1):
                ax.plot([nodes[i][0], nodes[i + 1][0]], [nodes[i][2], nodes[i + 1][2]],
                            color=color, alpha=alpha)
        else:
            projection_mat = simrpy2mat(proj_ea)
            nodes_proj = (np.dot(projection_mat, nodes.T)).T
            nodes_proj[:,1]=0
            # draw_ax.scatter(nodes_proj[1:, 0], nodes_proj[1:, 2], color=color, alpha=alpha)
            for i in range(nodes_proj.shape[0] - 1):
                ax.plot([nodes_proj[i][0], nodes_proj[i + 1][0]], [nodes_proj[i][2], nodes_proj[i + 1][2]],
                            color=color, alpha=alpha)
        return ax

    def sample_view(self, phi_xz, phi_y):
        """
        Parameters:
            phi_xz {float} -- [0,90], sample interval in x-z plane
            phi_y {float} -- [0,360], sample interval around y axis
        Returns:
            proj_angles {list} -- sampled view number
        """
        proj_angles = []
        for xz in np.arange(0,91,phi_xz):
            for y in np.arange(0,360,phi_y):
                if xz != 0:
                    # euler angle in 'xyz' seq
                    proj_angles.append([xz,y,0])
        return proj_angles

    def get_obj_height(self, nodes, proj_angle=None):
        """
        Compute average height for one object represented by node
        Arguments:
            nodes {array} -- shape=(node num x 3)
            proj_angle {list} -- 
        Return:
            {float} -- average height of this object
        """
        if proj_angle is None:
            return np.mean(nodes[:1])
        else:
            new_nodes = (np.dot(simrpy2mat(proj_angle), nodes.T)).T
            return np.mean(new_nodes[:1])

    def compute_writhe(self, graph):
        """
        Compute writhe value between multiple same objects
        Parameters:
            graph {array} -- (obj num x node num x 3)
        Returns:
            {float} -- sum of all gli
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
                    gli_sum += self.gli(seg1, seg2)
                obj_wmat[i][j] = gli_sum
                obj_wmat[j][i] = gli_sum
        # gli_mat = gli_mat + gli_mat.T
        return obj_wmat.sum(axis=1)
    
    def compute_writhe_matrix(self, graph, i, j):
        """
        Compute writhe matrix between object i and j
        Parameters:
            graph {array} -- (obj num x node num x 3)
            i ,j {int} -- obj index in graph
        Returns:
            gli_mat {array} -- (obj num x obj num) writhe matrix
        """
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
                gli = self.gli(obj1[k], obj2[t])
                gli_mat[k][t] = gli
        return gli_mat
    
    def compute_writhe_matrix(self, node1, node2):
        """
        Compute writhe matrix between two arbitrary objects
        Parameters:
            node1 {array} -- (node num1, 3), nodes of first object
            node2 {array} -- (node num2, 3), nodes of second object
        Returns:
            wmat [array] -- (node num1 x node num2), writhe matrix
        """
        edge1 = self.node2edge(node1)
        edge2 = self.node2edge(node2)

        n1_seg = edge1.shape[0]
        n2_seg = edge2.shape[0]
        wmat = np.zeros((n1_seg, n2_seg))
        for i in range(n1_seg):
            for j in range(n2_seg):
                gli = self.gli(edge1[i], edge2[j])
                wmat[i][j] = gli
        return wmat


    def transform_sim_pos(self, node, p, q):
        """
        Transfrom the template in model frame to a given pose
        * for new data, (p,q) is the pose of the first cube, no need to add trans_
        Parameters: 
            node {array} -- shape=(node num x 3), template node in object frame 
            p {array} -- shape=(3), pose of the first cube in sim frame 
            p {array} -- shape=(4), pose of the first cube in sim frame 
        Return:
            new_node {array} -- shape=(node num x 3)
        """
        
        m = quat2mat(q)
        # trans = -self.cube1
        # trans_ = m.dot(trans)
        # p += trans_
        return rotate_3d(node, m) + p

    def create_sim_graph(self, pose):
        """
        Using 6D poses to create the same scene in sim frame
        Arguments:
            pose {array} -- shape=(obj num x 7)
        Return:
            graph {list} -- len=(obj num:(node num x 3))
        """
        graph = []
        for v in pose:
            node = self.template.copy()
            p = v[0:3]
            q = v[3:7]
            node = self.transform_sim_pos(node, p, q)
            graph.append(node)
        return graph

    def compute_tangleship(self, graph, proj_angle=[0,0,0]):
        """
        Compute height among multiple same objects
        Arguments:
            graph {array} -- (obj num x node num x 3)
            pose {array}  -- (obj num x 6)
            proj_angle {list} -- [roll, pitch, yaw] with degrees
        """
        graph = np.array(graph)

        n_obj = graph.shape[0]
        # n_other_obj = graph.shape[0] - 1
        n_seg = graph.shape[1]-1
        print("[*] Object number is {} and segment number is {}".format(n_obj, n_seg))
        rot_graph = []
        objs = []
        objs_proj = []

        crossings = {}
        crossings["label"] = {}
        crossings["point"] = {}
        crossings["obj"] = {}

        # gli_mat = np.zeros((n_obj, n_obj))
        # check if projection neede
        if proj_angle != [0,0,0]:
            print(f"[*] Start projecting along {proj_angle} ...")
        
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
            crossings["label"][i] = []
            crossings["point"][i] = []
            crossings["obj"][i] = []
            for j in range(n_obj):
                for (k, t) in list(itertools.product(range(n_seg), range(n_seg))):
                    seg1_proj, seg2_proj = objs_proj[i][k], objs_proj[j][t]
                    seg1, seg2 = objs[i][k], objs[j][t]

                    gli = self.gli(seg1_proj, seg2_proj)
                    if gli*2 == 1:
                        crossing = calc_intersection([seg1[0], seg1[2]], [seg1[3], seg1[5]], 
                                                [seg2[0], seg2[2]], [seg2[3], seg2[5]])
                        # label the crossings for obj i - seg1, seg2, crossing 
                        d1 = calc_lineseg_dist([crossing[0],0,crossing[1]], seg1)
                        d2 = calc_lineseg_dist([crossing[0],0,crossing[1]], seg2)
                        if d1 > d2:
                            crossings["label"][i].append(1)
                            crossings["point"][i].append([crossing[0], d1, crossing[1]])
                            crossings["obj"][i].append(j)
                        elif d1 < d2:
                            crossings["label"][i].append(-1)
                            crossings["point"][i].append([crossing[0], d2, crossing[1]])
                            crossings["obj"][i].append(j)
        return crossings
