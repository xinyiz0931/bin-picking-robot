import os
import sys
# execute the script from the root directory etc. ~/src/myrobot
sys.path.append("./")
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
    sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
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
    r = R.from_euler('xyz', Re, degrees=True)
    return r.as_quat()

def rpy2mat(Re):
    r = R.from_euler('xyz', Re, degrees=True)
    return r.as_matrix()
    
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

def calc_intersection(a1, a2, b1, b2):
    """ 
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1,a2,b1,b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1)))) # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return (x/z, y/z)

def replace_bad_point(imgpath, loc, bounding_size=10):
    """    
    find the closest meaningful point for top-ranked grasp point
    to avoid some cases where the pixel value is zero or very low
    if the pixel value is 0 or 255, replace
    else return input values
    """
    
    (x, y) = loc
    gray = cv2.imread(imgpath, 0)
    background_pixel = 10
    if gray[y,x] < background_pixel: 
        h, w = gray.shape
        mat = gray[(y-bounding_size):(y+bounding_size+1),
                (x-bounding_size):(x+bounding_size+1)]
        left_margin = x-bounding_size
        top_margin = y-bounding_size
        max_xy = np.where(mat == mat.max())
        y_p = max_xy[0][0] + y-bounding_size
        x_p = max_xy[1][0] + x-bounding_size

        return 1, (x_p, y_p)
    else:
        return 0, loc
    


def camera_to_robot(c_x, c_y, c_z, angle, calib_path):
    """Using 4x4 calibration matrix to calculate x and y
        Using pixel value to calculate z
    """

    # get calibration matrix 4x4
    calibmat = np.loadtxt(calib_path)
    camera_pos = np.array([c_x, c_y, c_z, 1])
    r_x, r_y, r_z, _ = np.dot(calibmat, camera_pos)  # unit: mm --> m

    r_a = 180.0 * angle / math.pi
    if(r_a < -90):
        r_a = 180 + r_a
    elif(90 < r_a):
        r_a = r_a - 180
    # if(r_x >= 0.7):
    #     raise Exception("x value is incorrect! ")
    # if(r_y <= -0.3 or r_y >= 0.3):
    #     raise Exception("y value is incorrect! ")

    # r_z = r_z - 0.015

    # if(r_z <= 0.001 or r_z == 0):
    #     print("z value is incorrect but it can reach the table...")
    #     r_z = 0.001

    # r_z need to be revised
    return r_x, r_y, r_z, r_a


def image_to_robot(img_x, img_y, img_z, angle):

    # get calibration elements
    # [calib_sx, calib_sy, calib_tx, calib_ty] = self.calib_mat

    # robot_x = calib_sy*img_y + calib_ty
    # robot_y = calib_sx*img_x + calib_tx

    # mat = np.array([[0.00054028,0.0000389,0.04852181],[-0.00001502,0.00053068,-0.5952836],[0,0,1]])
    print("input: ", (1544-img_y), (2064-img_x))
    robot_frame = mat.dot([(1544-img_y), (2064-img_x), 1])
    robot_x = robot_frame[0]
    robot_y = robot_frame[1]

    # img_z_refined, robot_z =  self.height_heuristic(int(img_x), int(img_y))
    robot_z = 0.005 + 0.00045*(img_z - 50) - 0.01

    if(0.67 <= robot_x):
        print("Error: X座標が範囲外です")
    if(0.67 <= robot_y):
        print("Error: Y座標が範囲外です")
    # if(0.005152941176470586 >= robot_z):
    if(0.005 >= robot_z):
        print("Error: Y座標が範囲外です")
        robot_z = 0.005

    robot_angle = 180.0 * angle / math.pi
    if(robot_angle < -90):
        robot_angle = 180 + robot_angle
    elif(90 < robot_angle):
        robot_angle = robot_angle - 180

    print("\n---------------- Image ----------------")
    print("X : {}\nY : {}\nZ : {}\nA：{}".format(img_x, img_y, img_z, angle))
    print("\n---------------- Robot ----------------")
    print("X : {}\nY : {}\nZ : {}\nA：{}".format(
        robot_x, robot_y, robot_z, robot_angle))
    return robot_x, robot_y, robot_z, robot_angle


def rotate_point_cloud(pc, angle=-11.75):
    rad = np.radians(angle)
    # build trainsformation matrix 3x3
    H = np.array([
        [math.cos(rad),  0, math.sin(rad)],
        [0,              1, 0            ],
        [-math.sin(rad), 0, math.cos(rad)]
    ])
    return np.dot(H, pc.T).T


# print(rpy2quat([0,180,90]))