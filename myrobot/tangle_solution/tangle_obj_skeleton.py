"""
A python scripts to generate collision files for simulation models
Arthor: Xinyi
Date: 2021/10/15
---
Input: thinned skeleton of simulation model
Output: decomposed cubes for collision checking
---
Be careful when the position of each cube are not change, 
but the size of each cube are 1/2 of the original size. 
"""
import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
from myrobot.utils import *

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
    
    # def calc_center_of_mass(self, vertices):
    #     """
    #     vertices: shape=(N,8,3)
    #             N is the number of fitted cube, 8 is the number of vertices for cubes
    #     """
    #     vertices = np.array(vertices)
    #     print(vertices.shape)
    #     vnum = vertices.shape[0]*vertices.shape[1]
    #     vpoints = np.reshape(vertices, (vnum, 3))

    #     vsum = np.zeros(3)
    #     for p in vpoints:
    #         vsum += p
    #     return vsum/vnum

    def write_decomposed_obj(self, vertices, wrl_path):
        """
        vertices: shape=(N,8,3)
                N is the number of fitted cube, 8 is the number of vertices for cubes
        """
        vertices = np.array(vertices)
        vnum = vertices.shape[0]*vertices.shape[1]
        fp = open(wrl_path, 'wt')
        for v in vertices:
            print("#VRML V2.0 utf8\n", file=fp)
            print("Group {", file=fp)
            print("	children [", file=fp)
            print("		Shape {", file=fp)
            print("			appearance Appearance {", file=fp)
            print("				material Material {", file=fp)
            print("					diffuseColor 0.410000 0.670000 0.340000", file=fp)
            print("					ambientIntensity 0.400000", file=fp)
            print("					specularColor 0.500000 0.500000 0.500000", file=fp)
            print("					emissiveColor 0.000000 0.000000 0.000000", file=fp)
            print("					shininess 0.400000", file=fp)
            print("					transparency 0.000000", file=fp)
            print("				} ", file=fp)
            print("			} ", file=fp)
            print("			geometry IndexedFaceSet {", file=fp)
            print("				ccw TRUE", file=fp)
            print("				solid TRUE", file=fp)
            print("				convex TRUE", file=fp)
            print("				coord DEF co Coordinate {", file=fp)
            print("					point [", file=fp)
            for i in range(8):
                print(f"						{v[i][0]} {v[i][1]} {v[i][2]}", file=fp)

            print("					]", file=fp)
            print("				}", file=fp)
            print("				coordIndex [ ", file=fp)
            print("              0,  3,  2,  1, -1,", file=fp)
            print("              4,  5,  6,  7, -1,", file=fp)
            print("              0,  1,  5,  4, -1,", file=fp)
            print("              3,  0,  4,  7, -1,", file=fp)
            print("              2,  3,  7,  6, -1,", file=fp)
            print("              1,  2,  6,  5, -1,", file=fp)
            print("				]", file=fp)
            print("			}", file=fp)
            print("		}", file=fp)
            print("	]", file=fp)
            print("}\n", file=fp)

        fp.close()
        main_proc_print("Wrote decomposed object to a wrl file... ")
        return

    def decompose_obj(self, graph, de_obj_path):
        """Object skeleton ==> decompsed cubes 
           *Important! all sizes are shrinked to 1/2 in bin simulator
        Arguments:
            graph {dict} -- dictionary to describe an object's skeleton in x-y plane
        """
        node = np.array(graph["node"])
        edge = graph["edge"]

        cubes_size = []
        cubes_position = []
        cubes_quat = []

        fp = open(de_obj_path, 'wt')

        if graph["section"]["shape"] == "circle":
            radius = graph["section"]["size"] 
            cx = radius
            cz = radius
            
        elif graph["section"]["shape"] == "rectangle":
            cx, cz = graph["section"]["size"]

        cx, cz = cx/2, cz/2

        # fit cube to each edge
        cube_num = len(edge)

        # center of mass
        com = self.calc_center_of_mass(graph)
        
        print(" ".join(str(c_) for c_ in com), file=fp)
        print(f"{cube_num}",file=fp)

        cube_primitive = [0,1,0]
        cube_quat_primitive = [0,0,0,1]

        vertices = []
        for (i,j) in edge:
            vertex = []
            # 1. pre cube size
            cy = calc_2points_distance(node[i], node[j])/2
            # 2. cube position
            cp = (node[i]+node[j])/2
            # TODO: calculate vertices

            # 3. cube rotation
            angle_ =  calc_2vectors_angle(cube_primitive, np.abs(node[j] - node[i]))
            if angle_== 0:
                cq = cube_quat_primitive # y-axis rotate 45
                c_rot_mat = quat2mat(cube_quat_primitive)
            else:
                c_rot_mat = calc_2vectors_rot_mat(cube_primitive, (node[j] - node[i]))
                cq = mat2quat(c_rot_mat)

            ncx, ncy, ncz = np.dot(c_rot_mat,[cx, cy, cz])

            vertex.append([cp[0]-ncx, cp[1]+ncy, cp[2]-ncz])
            vertex.append([cp[0]-ncx, cp[1]-ncy, cp[2]-ncz])
            vertex.append([cp[0]+ncx, cp[1]-ncy, cp[2]-ncz])
            vertex.append([cp[0]+ncx, cp[1]+ncy, cp[2]-ncz])
            
            vertex.append([cp[0]-ncx, cp[1]+ncy, cp[2]+ncz])
            vertex.append([cp[0]-ncx, cp[1]-ncy, cp[2]+ncz])
            vertex.append([cp[0]+ncx, cp[1]-ncy, cp[2]+ncz])
            vertex.append([cp[0]+ncx, cp[1]+ncy, cp[2]+ncz])


            vertices.append(vertex)
            # save the data
            # cubes_size.append([cx,cy,cz])
            # cubes_position.append(cp)
            # cubes_quat.append(cq)

            # write to file
            print(f"{cx} {cy} {cz}",file=fp)
            print(" ".join(str(p_) for p_ in cp), file=fp)
            print(f"{cq[3]} {cq[0]} {cq[1]} {cq[2]}", file=fp)
            print(" ".join(str(r_) for r_ in c_rot_mat[0]), file=fp)
            print(" ".join(str(r_) for r_ in c_rot_mat[1]), file=fp)
            print(" ".join(str(r_) for r_ in c_rot_mat[2]), file=fp)

        fp.close()
        main_proc_print("Wrote collision file... ")

        return vertices

def from_ske():
    """Manually write the object graph to generate json file"""
    shape = "cc"

    write_path = f"./objmodel\\skeleton_{shape}.json"
    collision_path = f"./objmodel\\collision_{shape}.txt"
    wrl_path = f"./objmodel\\cube_{shape}.wrl"
    # ================ obj info ====================
    obj_ske = {}
    obj_ske["node"] = [[-40,-30,0], [-37.321, -40, 0], [-30, -47.321, 0], [-20, -50, 0],
                       [-10,-47.321, 0], [-2.679,-40,0],[0,-30,0],
                       [0, 40, 0], [0, 40, 30]]
    obj_ske["edge"] = [[0,1], [1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]
    obj_ske["section"] = {}
    obj_ske["section"]["shape"] = "circle"
    obj_ske["section"]["size"] = 7
    # graph["section"]["shape"] = "rectangle"
    # graph["section"]["size"] = [2, 10]
    # ================ obj info ====================

    tok = TangleObjSke()
    tok.write_obj(write_path, obj_ske)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tok.draw_obj_skeleton(obj_ske, ax, "green")

    center = tok.calc_center_of_mass(obj_ske)
    ax.scatter(center[0], center[1], center[2], color='red')

    vertices=tok.decompose_obj(obj_ske, collision_path)
    
    for v in vertices:
        tok.draw_node(v, ax, 'blue', alpha=0.5)

    plt.show()

def from_obj():
    shape = "cc"

    # write_path = f"D:\\code\\myrobot\\objmodel\\skeleton_{shape}.json"
    # collision_path = f"D:\\code\\myrobot\\objmodel\\collision_{shape}.txt"

    skeleton_path = f"./objmodel\\skeleton_{shape}.json"
    collision_path = f"./objmodel\\collision_{shape}.txt"
    wrl_path = f"./objmodel\\cube_{shape}.wrl"

    skeleton_path = f"./objmodel/skeleton_{shape}.json"
    collision_path = f"./objmodel/collision_{shape}.txt"
    wrl_path = f"./objmodel/cube_{shape}.wrl"


    tok = TangleObjSke()
    obj_ske = tok.load_obj(skeleton_path)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tok.draw_obj_skeleton(obj_ske, ax, "green")

    center = tok.calc_center_of_mass(obj_ske)
    ax.scatter(center[0], center[1], center[2], color='red')

    vertices=tok.decompose_obj(obj_ske, collision_path)

    # tok.write_decomposed_obj(vertices, wrl_path)
    
    for v in vertices:
        tok.draw_node(v, ax, 'blue', alpha=0.5)

    # optional: write vertices to the vertex_*.txt
    plt.show()

if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    # from_ske()
    from_obj()

    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))


