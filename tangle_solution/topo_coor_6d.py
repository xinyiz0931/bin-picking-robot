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
from sklearn import svm
np.set_printoptions(suppress=True)
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from scipy import ndimage
from utils.base_utils import *
from utils.image_proc_utils import *
from utils.plot_utils import *
            
class TopoCoor6D(object):
    def __init__(self, this=0):
        self.this = this

    def cross_product(self, a, b):
        x1, y1, z1 = a
        x2, y2, z2 = b
        return np.array([(y1 * z2 - y2 * z1), -(x1 * z2 - x2 * z1), (x1 * y2 - x2 * y1)])

    def gli_original(self, line1, line2):
        """
        Calculate Gaussian link integral in a simple way
        Input: two line segment line = [p1.x,p1.y.p1.z, p2.x,p2.y.p2.z]
        output: w [float]
        """
        a = np.array([line1[0], line1[1], line1[2]])
        b = np.array([line1[3], line1[4], line1[5]])

        c = np.array([line2[0], line2[1], line2[2]])
        d = np.array([line2[3], line2[4], line2[5]])

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

        w = np.arcsin(np.dot(n_a, n_b)) + np.arcsin(np.dot(n_b, n_c)) + np.arcsin(np.dot(n_c, n_d)) + np.arcsin(
            np.dot(n_d, n_a))
        return np.nan_to_num(w)
    
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
    def draw_node(self, node, draw_ax, color, alpha=1, ):
        # color = tuple([x / 255 for x in color])
        draw_ax.scatter(node[:, 0], node[:, 1], node[:, 2], color=color, alpha=alpha)
        for i in range(node.shape[0] - 1):
            draw_ax.plot([node[i][0], node[i + 1][0]], [node[i][1], node[i + 1][1]], [node[i][2], node[i + 1][2]],
                        color=color, alpha=alpha)
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
        obj_collection = []
        obj_wmat = np.zeros((n_obj, n_obj))

        for nodes in graph:
            # obj number
            obj_collection.append(self.node2edge(nodes))
        obj_collection = np.array(obj_collection)

        for i in range(n_obj):
            obj1 = obj_collection[i]
            for j in range(i + 1, n_obj):
                gli_sum = 0
                # obj #i & obj #j
                obj2 = obj_collection[j]
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
        obj_collection = []
        for nodes in graph:
            # obj number
            obj_collection.append(self.node2edge(nodes))

        obj_collection = np.array(obj_collection)

        n_seg = obj_collection.shape[1]

        gli_mat = np.zeros((n_seg, n_seg))
        obj1 = obj_collection[i]
        obj2 = obj_collection[j]
        for k in range(n_seg):
            for t in range(n_seg):
                gli = self.gli_original(obj1[k], obj2[t])
                gli_mat[k][t] = gli
        return gli_mat

    def transfer_sim_pos(self, node, p, q, center, cube1_pos):
        trans = center - cube1_pos
        m = quaternion2mat(q)
        trans_ = np.array([np.dot(trans, m[0]), np.dot(trans, m[1]), np.dot(trans, m[2])])
        p += trans_
        return rotate_3d(node, m) + p

    def make_sim_graph(self, template, pose, center, cube1_pos):
        graph = []
        for v in pose:
            # make a new template
            node = template.copy()
            pos = v[0:3]
            qua = v[3:7]
            # translation P, rotation R
            rot = quaternion2mat(qua)
            P = pos - cube1_pos
            rot_ori = pos
            node = self.transfer_sim_pos(node, pos, qua, center, cube1_pos)
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
        n_seg = graph.shape[1]
        result_print("Object number is {} and segment number is {}".format(n_obj, n_seg))
        obj_collection = []

        gli_mat = np.zeros((n_obj, n_obj))

        for nodes in graph:
            # obj number
            obj_collection.append(self.node2edge(nodes))
        obj_collection = np.array(obj_collection)

        # get their heights
        avhglist = self.compute_avg_height(graph)
        for i in range(n_obj):
            obj1 = obj_collection[i]
            for j in range(i + 1, n_obj):
                gli_sum = 0
                # obj #i & obj #j
                obj2 = obj_collection[j]
                for (seg1, seg2) in list(itertools.product(obj1, obj2)):
                    gli_sum += self.gli_original(seg1, seg2)
                gli_sum = gli_sum / n_seg
                gli_mat[i][j] = gli_sum
                # gli_mat[j][i] = gli_sum
                distance = np.linalg.norm(pose[j] - pose[i])
                diff_h = np.abs(avhglist[i] - avhglist[j])
                singulated = self.check_overlap(root_dir, i, j)
                # print("obj {}&{}: writhe={:.3f}, singulated?: {}, height=({:.3f}, {:.3f})".format(i, j, gli_sum, singulated, avhglist[i], avhglist[j]))

        """start voting"""
        voting = np.zeros(n_obj)
        eliminate_mat = gli_mat
        vrow = np.array(range(n_obj))
        pair_list = list(itertools.combinations(range(n_obj), 2))
        gli_thre = 0.5

        random.shuffle(pair_list)
        for p in pair_list:
            i, j = p
            main_proc_print("Analyzing obj ({}, {}): gli = {:.3f} ...".format(i, j, gli_mat[i][j]))
            overlapped = self.check_overlap(root_dir, i, j)  # true for overlap
            if gli_mat[i][j] >= gli_thre:
                if overlapped == True:
                    voting[i] -= 1
                    voting[j] -= 1
                else:
                    if avhglist[i] > avhglist[j]:
                        voting[j] -= 1
                    else:
                        voting[i] -= 1
            else:
                if overlapped == True:
                    if avhglist[i] > avhglist[j]:
                        voting[j] -= 1
                    else:
                        voting[i] -= 1
                # else:
                #     if avhglist[i] > avhglist[j]:
                #         voting[j] -= 1
                #     else:
                #         voting[i] -= 1
        result_print(f"Current voting list: {voting}")
        glilist = (gli_mat + gli_mat.T).sum(axis=1)
        if (voting < 0).all():
            """all < 0 -> select from writhe list, including dragging/sliding"""
            pick_obj = np.argmin(glilist)
            main_proc_print(f"Check obj.{pick_obj} tangleship...")
            pair_list_move = []
            wmat_list = []
            wval_list = []
            for i in vrow:
                if i != pick_obj: pair_list_move.append((pick_obj, i))

            for p in pair_list_move:
                (i, j) = p
                wmat = self.compute_writhe_matrix(graph, i, j)
                wmat_list.append(wmat)
                wval_list.append(np.sum(wmat))

            # visualize all writhe matrix of obj i
            fig2 = plot_subfigures(wmat_list, max_ncol=5)
            (_, avoid_obj) = pair_list_move[wval_list.index(max(wval_list))]
            result_print(f"Drag obj.{pick_obj} far away from obj.{avoid_obj}...")
        else:
            """one obj has no negative voting, picking it up"""
            pick_obj = vrow[voting == 0][0]
            print(f"Pick obj.{pick_obj}")

        # for (i,j) in itertools.combinations(range(n_obj),2):
        #     print(i,j)

        return glilist, avhglist
    
    def find_writhe_thre(self, glilist, avhglist):
        X = glilist.reshape(-1,1)
        Y = [int(h) for h in avhglist]
    
        clf = svm.SVC(decision_function_shape='ovo')
        clf.fit(X, Y)
        dec = clf.decision_function([[1]])
        num_class = dec.shape[1] # 4 classes: 4*3/2 = 6
        plt.scatter(glilist, avhglist, marker='o')
        plt.show()
        return

