from termcolor import colored, cprint
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

"""
All quaternions are formatted as `(x,y,z,w)`
Pay attention with the phycics simulator
"""
###########################################################################
################################# MATH UTILS ##############################
###########################################################################

def quat2mat(Rq):
    r = R.from_quat(Rq)
    return r.as_matrix()


def mat2quat(Rm):
    r = R.from_matrix(Rm)
    return r.as_quat()

def quat2rpy(Rq):
    # unit: degree
    r = R.from_quat(Rq)
    return r.as_euler('xyz', degrees=True)

def rpy2quat(Re):
    r= R.from_euler('xyz', Re, degrees=True)
    return r.as_quat()

    
def rotate_3d(p, R, origin=(0, 0, 0)):
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)

def calc_2points_distance(p1, p2):
    return np.linalg.norm(p1-p2)

def calc_2vectors_angle(v1, v2):
    """Calculate the angle between v1 and v2
       Returns: angles in degree
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    dot_product = np.dot(unit_v1, unit_v2)
    return np.rad2deg(np.arccos(dot_product))

def calc_2vectors_rot_mat(v1, v2):
    """Calculate rotation matrix between two vectors v1, v2
    Arguments:
        v1 -- source vector
        v2 -- destination vector
    Returns:
        rotation_matrix
    """
    v1 = np.array(v1)
    v2 = np.array(v2)
    unit_v1 = v1 / np.linalg.norm(v1)
    unit_v2 = v2 / np.linalg.norm(v2)
    v = np.cross(unit_v1, unit_v2)
    c = np.dot(unit_v1, unit_v2)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix

###########################################################################
############################ PRINT UTILS ##################################
###########################################################################
def main_proc_print(main_proc_str):
    (lambda x: cprint(x, 'yellow'))("[ MAIN PROCESS ] "+str(main_proc_str))
    

def warning_print(warning_str):
    (lambda x: cprint(x, 'red'))("[   WARNINGS   ] "+str(warning_str))

def result_print(result_str):
    (lambda x: cprint(x, 'green'))("[    OUTPUT    ] "+str(result_str))


def important_print(result_str):
    (lambda x: cprint(x, 'white',  'on_green'))("[    NOTICE    ] "+str(result_str))


