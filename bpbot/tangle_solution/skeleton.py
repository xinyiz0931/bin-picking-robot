"""
A python scripts to generate collision files for simulation models
Arthor: Xinyi
Date: 2021/10/15
"""
import os
import sys
import json
from venv import create
import numpy as np
import matplotlib.pyplot as plt
from bpbot.utils import *

class TangleObjSke(object):
    def __init__(self):
        pass
    
    def load_obj_from_json(self, json_path):
        """
        Load graph from object's json file
        """
        try:
            with open(json_path) as jf:
                graph = json.load(jf)
            jf.close()
            return graph
        except IOError:
            warn_print("Not a json file")

    def write_obj_to_json(self, json_path, graph):
        """
        Write revised graph to object's json file
        """
        try:
            with open(json_path, 'w', encoding='utf-8') as jf:
                json.dump(graph, jf, ensure_ascii=False)
            main_print("Wrote to json file")    
            jf.close()
        except IOError:
            warn_print("Fail to write")
      
    def draw_obj_skeleton(self, graph, ax, color, alpha=1):
        """
        Draw one object (graph=nodes+edges) in 2D space
        """
        node = np.array(graph["node"])
        ax.scatter(node[:, 0], node[:, 1], node[:, 2], color=color, alpha=alpha)
        edge = graph["edge"]
        for (i,j) in edge:
            ax.plot([node[i][0], node[j][0]], [node[i][1], node[j][1]], [node[i][2], node[j][2]],
                color=color, alpha=alpha)
    
    def draw_node(self, node, ax, color, alpha=1):
        """
        Draw nodes in 2D space
        """
        node = np.array(node)
        ax.scatter(node[:, 0], node[:, 1], node[:, 2], color=color, alpha=alpha)
        # ax.scatter(node[0][0], node[0][1], node[0][2], color='r', alpha=1)
        # ax.scatter(node[1][0], node[1][1], node[1][2], color='g', alpha=1)
        # ax.scatter(node[2][0], node[2][1], node[2][2], color='b', alpha=1)
        # ax.scatter(node[3][0], node[3][1], node[3][2], color='y', alpha=1)

        # ax.scatter(node[4][0], node[4][1], node[4][2], color='c', alpha=0.3)
        # ax.scatter(node[5][0], node[5][1], node[5][2], color='m', alpha=0.3)
        # ax.scatter(node[6][0], node[6][1], node[6][2], color='k', alpha=0.3)
        # ax.scatter(node[7][0], node[7][1], node[7][2], color='pink', alpha=0.3)

    def show_ply(self, ply_path):
        import open3d as o3d
        mesh = o3d.io.read_triangle_mesh(ply_path)
        mesh.compute_vertex_normals()
        print(np.asarray(mesh.triangle_normals))
        o3d.visualization.draw_geometries([mesh])

    def calc_center_of_mass(self, graph):
        node = np.array(graph["node"])
        edge = graph["edge"]
        edge_num = len(edge)
        vertice_sum = np.zeros(3)
        for (i,j) in edge:
            vertice_sum += node[i]
            vertice_sum += node[j]
        return vertice_sum / (edge_num*2)
    
    def decompose_obj_to_cube(self, graph, de_obj_path):
        """
        Object skeleton (graph) ==> decompsed cubes 
        * Cube: position of center, size (l,w,h) from center
        Parameters:
            graph {dict} -- "node": [[x,y,z],...], "edge": [[0,1],...], 
                         -- "section": {"shape":"circle", "size":10}
            de_obj_path {str} -- `collision_shape.txt` for decomposed cubes
                              -- must be the same as the one used in simulator
        Returns: 
            vertices {array} -- (fitted cube num,8,3)
        """
        node = np.array(graph["node"])
        edge = graph["edge"]

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
        
        fp = open(de_obj_path, 'wt')
        print(" ".join(str(c_) for c_ in com), file=fp)
        print(f"{cube_num}",file=fp)

        cube_primitive = [0,1,0]
        cube_quat_primitive = [0,0,0,1]

        vertices = []
        rot_vertice = []
        for (i,j) in edge:
            vertex = []
            # 1. pre cube size
            cy = calc_2points_distance(node[i], node[j])/2
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

            
            ncx, ncy, ncz = np.dot(c_rot_mat,[cx, cy, cz])
            ncp = np.dot(c_rot_mat,[cx, cy, cz])

            # vertex.append([cp[0]-ncx, cp[1]+ncy, cp[2]-ncz])
            # vertex.append([cp[0]-ncx, cp[1]-ncy, cp[2]-ncz])
            # vertex.append([cp[0]+ncx, cp[1]-ncy, cp[2]-ncz])
            # vertex.append([cp[0]+ncx, cp[1]+ncy, cp[2]-ncz])
            
            # vertex.append([cp[0]-ncx, cp[1]+ncy, cp[2]+ncz])
            # vertex.append([cp[0]-ncx, cp[1]-ncy, cp[2]+ncz])
            # vertex.append([cp[0]+ncx, cp[1]-ncy, cp[2]+ncz])
            # vertex.append([cp[0]+ncx, cp[1]+ncy, cp[2]+ncz])

            # vertex.append([cp[0]-cx, cp[1]+cy, cp[2]-cz])
            # vertex.append([cp[0]-cx, cp[1]-cy, cp[2]-cz])
            # vertex.append([cp[0]+cx, cp[1]-cy, cp[2]-cz])
            # vertex.append([cp[0]+cx, cp[1]+cy, cp[2]-cz])
            # vertex.append([cp[0]-cx, cp[1]+cy, cp[2]+cz])
            # vertex.append([cp[0]-cx, cp[1]-cy, cp[2]+cz])
            # vertex.append([cp[0]+cx, cp[1]-cy, cp[2]+cz])
            # vertex.append([cp[0]+cx, cp[1]+cy, cp[2]+cz])
            
            vertex.append(np.dot(c_rot_mat,[0-cx, 0+cy, 0-cz])+cp)
            vertex.append(np.dot(c_rot_mat,[0-cx, 0-cy, 0-cz])+cp)
            vertex.append(np.dot(c_rot_mat,[0+cx, 0-cy, 0-cz])+cp)
            vertex.append(np.dot(c_rot_mat,[0+cx, 0+cy, 0-cz])+cp)
            vertex.append(np.dot(c_rot_mat,[0-cx, 0+cy, 0+cz])+cp)
            vertex.append(np.dot(c_rot_mat,[0-cx, 0-cy, 0+cz])+cp)
            vertex.append(np.dot(c_rot_mat,[0+cx, 0-cy, 0+cz])+cp)
            vertex.append(np.dot(c_rot_mat,[0+cx, 0+cy, 0+cz])+cp)
            
            vertex.append(cp)
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
        main_print("Wrote collision file... ")
        return vertices

    def write_decomposed_to_wrl(self, vertices, wrl_path):
        """
        Write the decomposed cubes vertices to a wrl file, unit: m
        Parameters:
            vertices {array} -- (fitted cube num,8,3)
            wrl_path {str} 
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
                print(f"						{v[i][0]/1000} {v[i][1]/1000} {v[i][2]/1000}", file=fp)

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
        
        # 0,  3,  2,  1, -1,
	    # 4,  5,  6,  7, -1,
	    # 0,  1,  5,  4, -1,
	    # 3,  0,  4,  7, -1,
	    # 2,  3,  7,  6, -1,
	    # 1,  2,  6,  5, -1,

        fp.close()
        main_print("Wrote decomposed object to a wrl file... ")

def create_obj():
    """
    1. Manually define a new object (shape, node, edge...)
    2. Create the json file for object
    3. Decompose this object using cubes and write to the collision.txt file
    4. Visualize
    """
    shape = "ed"
    obj_json_path = os.path.join("./objmodel", f"skeleton_{shape}.json")
    collision_path = os.path.join("./objmodel", f"collision_{shape}.txt")
    wrl_path = os.path.join("./objmodel", f"cube_{shape}.wrl")

    # 1. ===================================================
    obj_ske = {}
    # obj_ske["node"] = [[-28, 20, 0], [-28, 49, 0], [11, 49, 0], [11, -49, 0], [11, -49, 39], [11, -20, 39]]
    obj_ske["node"] = [[0,-79,0],[0,-19,0],[-9.5,-16.454,0],[-16.454,-9.5,0],[-19,0,0],[-16.454,9.5,0],[-9.5,16.454,0],[0,19,0],[9.5,16.454,0],[16.454,9.5,0],[19,0,0],[16.454,-9.5,0],[9.5,-16.454,0]]
    obj_ske["edge"] = [[0,1],[1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8],[8,9],[9,10],[10,11],[11,12],[12,1]]
    # obj_ske["edge"] = [[0,1],[1,2],[2,3],[3,4],[4,5]]
    # obj_ske["node"] = [[-40,-30,0], [-37.321, -40, 0], [-30, -47.321, 0], [-20, -50, 0],
    #                    [-10,-47.321, 0], [-2.679,-40,0],[0,-30,0],
    #                    [0, 40, 0], [0, 40, 30]]
    # obj_ske["edge"] = [[0,1], [1,2],[2,3],[3,4],[4,5],[5,6],[6,7],[7,8]]
    obj_ske["section"] = {}
    obj_ske["section"]["shape"] = "circle"
    obj_ske["section"]["size"] = 8
    # graph["section"]["shape"] = "rectangle"
    # graph["section"]["size"] = [2, 10]

    # 2. ===================================================
    tok = TangleObjSke()
    tok.write_obj_to_json(obj_json_path, obj_ske)

    # 3. ===================================================
    center = tok.calc_center_of_mass(obj_ske)
    vertices=tok.decompose_obj_to_cube(obj_ske, collision_path)

    # 4. ===================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tok.draw_obj_skeleton(obj_ske, ax, "green")
    for v in vertices:
        tok.draw_node(v, ax, 'blue', alpha=0.5)
    ax.scatter(center[0], center[1], center[2], color='red')
    plt.show()

def decompose_obj(shape):
    """
    1. Load a object from json file
    2. Decompose this object using cubes and write to the collision.txt file
    3. Visualize
    """

    obj_json_path = os.path.join("./objmodel", f"skeleton_{shape}.json")
    collision_path = os.path.join("./objmodel", f"collision_{shape}.txt")
    wrl_path = os.path.join("./objmodel", f"cube_{shape}.wrl")

    # 1. ===================================================
    tok = TangleObjSke()
    obj_ske = tok.load_obj_from_json(obj_json_path)

    # 2. ===================================================
    center = tok.calc_center_of_mass(obj_ske)
    vertices=tok.decompose_obj_to_cube(obj_ske, collision_path)
    # optional: write vertices to the vertex_*.txt
    tok.write_decomposed_to_wrl(vertices, wrl_path)

    # 3. ===================================================
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    tok.draw_obj_skeleton(obj_ske, ax, "green")
    ax.scatter(center[0], center[1], center[2], color='red')
    for v in vertices:
        # v = vertices[0]
        tok.draw_node(v, ax, 'blue', alpha=0.5)
        # tok.draw_node(v, ax, 'blue', alpha=1)
    plt.show()

if __name__ == "__main__":
    
    import timeit
    start = timeit.default_timer()

    tok = TangleObjSke()

    # tok.show_ply(ply_path="./objmodel\\model_cc.ply")
    
    # create_obj()
    # shapes = ["cc", "cr", "e", "eb", "f", "j", "sc", "sr", "st", "u"]
    # for shape in shapes:
    #     decompose_obj(shape)
    decompose_obj("st")

    end = timeit.default_timer()
    main_print("Time: {:.2f}s".format(end - start))
    

