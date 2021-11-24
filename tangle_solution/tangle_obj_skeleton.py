import os
import sys
sys.path.append("./")
import json
import numpy as np
import matplotlib.pyplot as plt
from utils.plot_utils import plot_subfigures, get_cmap
from utils.base_utils import *
from utils.transform_utils import *

class TangleObjSke(object):
    def __init__(self):
        pass
    
    def load_obj(self, obj_path):
        try:
            with open(obj_path) as jf:
                graph = json.load(jf)
            jf.close()
            return graph
        except IOError:
            warning_print("Not a json file")

    def write_obj(self, obj_path, graph):
        try:
            with open(obj_path, 'w', encoding='utf-8') as jf:
                json.dump(graph, jf, ensure_ascii=False)
            main_proc_print("Wrote to json file")    
            jf.close()
        except IOError:
            warning_print("Fail to write")
        
    def draw_obj_skeleton(self, graph, draw_ax, color, alpha=1):
        node = np.array(graph["node"])
        draw_ax.scatter(node[:, 0], node[:, 1], node[:, 2], color=color, alpha=alpha)
        edge = graph["edge"]
        for (i,j) in edge:
            draw_ax.plot([node[i][0], node[j][0]], [node[i][1], node[j][1]], [node[i][2], node[j][2]],
                color=color, alpha=alpha)

    def draw_node(self, node, draw_ax, color, alpha=1):
        node = np.array(node)
        draw_ax.scatter(node[:, 0], node[:, 1], node[:, 2], color=color, alpha=alpha)

    def calc_center_of_mass(self, graph):
        node = np.array(graph["node"])
        edge = graph["edge"]
        edge_num = len(edge)
        vertice_sum = np.zeros(3)
        for (i,j) in edge:
            vertice_sum += node[i]
            vertice_sum += node[j]
        return vertice_sum / (edge_num*2)
    
    def decompose_obj(self, graph, de_obj_path):
        """Object skeleton ==> decompsed cubes 
           *Important! all sizes are shrinked to 1/2 in bin simulator
        Arguments:
            graph {dict} -- dictionary to describe an object's skeleton in x-y plane
        """
        node = np.array(graph["node"])
        edge = graph["edge"]

        fp = open(de_obj_path, 'wt')

        if graph["section"]["shape"] == "circle":
            radius = graph["section"]["size"] 
            cx = radius
            cz = radius
        elif graph["section"]["shape"] == "rectangle":
            cx, cz = graph["section"]["size"] 
        # fit cube to each edge
        cube_num = len(edge)
        # center of mass
        com = self.calc_center_of_mass(graph)
        
        
        print(" ".join(str(c_) for c_ in com), file=fp)
        print(f"{cube_num}",file=fp)
        cube_primitive = [0,1,0]
        cube_quat_primitive = [0,0,0,1]

        for (i,j) in edge:
            # 1. pre cube size
            cy = calc_2points_distance(node[i], node[j])
            # 2. cube position
            cp = (node[i]+node[j])/2
            # 3. cube rotation
            angle_ =  calc_2vectors_angle(cube_primitive, np.abs(node[j] - node[i]))
            if angle_== 0:
                cq = cube_quat_primitive # y-axis rotate 45
                c_rot_mat = quat2mat(cube_quat_primitive)
            else:
                c_rot_mat = calc_2vectors_rot_mat(cube_primitive, (node[j] - node[i]))
                cq = mat2quat(c_rot_mat)
            # write to file
            print(f"{cx/2} {cy/2} {cz/2}",file=fp)
            print(" ".join(str(p_) for p_ in cp), file=fp)
            print(f"{cq[3]} {cq[0]} {cq[1]} {cq[2]}", file=fp)
            print(" ".join(str(r_) for r_ in c_rot_mat[0]), file=fp)
            print(" ".join(str(r_) for r_ in c_rot_mat[1]), file=fp)
            print(" ".join(str(r_) for r_ in c_rot_mat[2]), file=fp)

        fp.close()
        main_proc_print("Wrote collision file... ")
        

def main():
    shape = "scylindertwist"

    # write_path = f"D:\\code\\myrobot\\objmodel\\skeleton_{shape}.json"
    # collision_path = f"D:\\code\\myrobot\\objmodel\\collision_{shape}.txt"

    write_path = f"./objmodel\\skeleton_{shape}.json"
    collision_path = f"./objmodel\\collision_{shape}.txt"

    tok = TangleObjSke()
    obj_ske = tok.load_obj(write_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tok.draw_obj_skeleton(obj_ske, ax, "green")
    center = tok.calc_center_of_mass(obj_ske)
    ax.scatter(center[0], center[1], center[2], color='red')
    plt.show()

    tok.decompose_obj(obj_ske, collision_path)

if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    main()

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))


