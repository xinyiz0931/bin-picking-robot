from scipy.spatial.transform import Rotation as R
import matplotlib.colors as colors
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import matplotlib
from collections import namedtuple
import numpy as np
from termcolor import colored, cprint
import os
from pickletools import read_unicodestring1
import sys
from turtle import shape
import cv2
import math
import colorama
colorama.init()

# ============================ PRINT UTILS ============================


def main_print(proc_str):
    # (lambda x: cprint(x, 'white'))("[ MAIN PROCESS ] "+str(main_proc_str))
    print("[     MAIN     ] "+str(proc_str))


def warn_print(warning_str):
    (lambda x: cprint(x, 'red'))("[   WARNINGS   ] "+str(warning_str))


def notice_print(result_str):
    # (lambda x: cprint(x, 'green', attrs=['bold']))("[    OUTPUT    ] "+str(result_str))
    (lambda x: cprint(x, 'green'))("[    OUTPUT    ] "+str(result_str))
    # (lambda x: cprint(x, 'grey',  'on_green'))("[    NOTICE    ] "+str(result_str))


# ============================ PLOT UTILS ============================

def plot_values(V):
    # reshape value function
    V_sq = np.reshape(V, (4, 4))

    # plot the state-value function
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    im = ax.imshow(V_sq, cmap='cool')
    for (j, i), label in np.ndenumerate(V_sq):
        ax.text(i, j, np.round(label, 5),
                ha='center', va='center', fontsize=14)
    plt.tick_params(bottom='off', left='off',
                    labelbottom='off', labelleft='off')
    plt.title('State-Value Function')
    plt.show()


def plot_blackjack_values(V):
    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in V:
            return V[x, y, usable_ace]
        else:
            return 0

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(1, 11)
        X, Y = np.meshgrid(x_range, y_range)

        Z = np.array([get_Z(x, y, usable_ace)
                     for x, y in zip(np.ravel(X), np.ravel(Y))]).reshape(X.shape)

        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=plt.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.set_zlabel('State Value')
        ax.view_init(ax.elev, -120)

    fig = plt.figure(figsize=(20, 20))
    ax = fig.add_subplot(211, projection='3d')
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(212, projection='3d')
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


def plot_policy(policy):
    def get_Z(x, y, usable_ace):
        if (x, y, usable_ace) in policy:
            return policy[x, y, usable_ace]
        else:
            return 1

    def get_figure(usable_ace, ax):
        x_range = np.arange(11, 22)
        y_range = np.arange(10, 0, -1)
        X, Y = np.meshgrid(x_range, y_range)
        Z = np.array([[get_Z(x, y, usable_ace) for x in x_range]
                     for y in y_range])
        surf = ax.imshow(Z, cmap=plt.get_cmap('Pastel2', 2),
                         vmin=0, vmax=1, extent=[10.5, 21.5, 0.5, 10.5])
        plt.xticks(x_range)
        plt.yticks(y_range)
        plt.gca().invert_yaxis()
        ax.set_xlabel('Player\'s Current Sum')
        ax.set_ylabel('Dealer\'s Showing Card')
        ax.grid(color='w', linestyle='-', linewidth=1)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.1)
        cbar = plt.colorbar(surf, ticks=[0, 1], cax=cax)
        cbar.ax.set_yticklabels(['0 (STICK)', '1 (HIT)'])

    fig = plt.figure(figsize=(15, 15))
    ax = fig.add_subplot(121)
    ax.set_title('Usable Ace')
    get_figure(True, ax)
    ax = fig.add_subplot(122)
    ax.set_title('No Usable Ace')
    get_figure(False, ax)
    plt.show()


def plot_cost_to_go_mountain_car(env, estimator, num_tiles=20):
    x = np.linspace(
        env.observation_space.low[0], env.observation_space.high[0], num=num_tiles)
    y = np.linspace(
        env.observation_space.low[1], env.observation_space.high[1], num=num_tiles)
    X, Y = np.meshgrid(x, y)
    Z = np.apply_along_axis(
        lambda _: -np.max(estimator.predict(_)), 2, np.dstack([X, Y]))

    fig = plt.figure(figsize=(10, 5))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                           cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
    ax.set_xlabel('Position')
    ax.set_ylabel('Velocity')
    ax.set_zlabel('Value')
    ax.set_title("Mountain \"Cost To Go\" Function")
    fig.colorbar(surf)
    plt.show()


def plot_value_function(V, title="Value Function"):
    """
    Plots the value function as a surface plot.
    """
    min_x = min(k[0] for k in V.keys())
    max_x = max(k[0] for k in V.keys())
    min_y = min(k[1] for k in V.keys())
    max_y = max(k[1] for k in V.keys())

    x_range = np.arange(min_x, max_x + 1)
    y_range = np.arange(min_y, max_y + 1)
    X, Y = np.meshgrid(x_range, y_range)

    # Find value for all (x, y) coordinates
    Z_noace = np.apply_along_axis(
        lambda _: V[(_[0], _[1], False)], 2, np.dstack([X, Y]))
    Z_ace = np.apply_along_axis(
        lambda _: V[(_[0], _[1], True)], 2, np.dstack([X, Y]))

    def plot_surface(X, Y, Z, title):
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111, projection='3d')
        surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                               cmap=matplotlib.cm.coolwarm, vmin=-1.0, vmax=1.0)
        ax.set_xlabel('Player Sum')
        ax.set_ylabel('Dealer Showing')
        ax.set_zlabel('Value')
        ax.set_title(title)
        ax.view_init(ax.elev, -120)
        fig.colorbar(surf)
        plt.show()

    plot_surface(X, Y, Z_noace, "{} (No Usable Ace)".format(title))
    plot_surface(X, Y, Z_ace, "{} (Usable Ace)".format(title))


def plot_episode_stats(stats, smoothing_window=10, noshow=False):
    # Plot the episode length over time
    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(stats.episode_lengths)
    plt.xlabel("Episode")
    plt.ylabel("Episode Length")
    plt.title("Episode Length over Time")
    if noshow:
        plt.close(fig1)
    else:
        plt.show(fig1)

    # Plot the episode reward over time
    fig2 = plt.figure(figsize=(10, 5))
    rewards_smoothed = pd.Series(stats.episode_rewards).rolling(
        smoothing_window, min_periods=smoothing_window).mean()
    plt.plot(rewards_smoothed)
    plt.xlabel("Episode")
    plt.ylabel("Episode Reward (Smoothed)")
    plt.title("Episode Reward over Time (Smoothed over window size {})".format(
        smoothing_window))
    if noshow:
        plt.close(fig2)
    else:
        plt.show(fig2)

    # Plot time steps and episode number
    fig3 = plt.figure(figsize=(10, 5))
    plt.plot(np.cumsum(stats.episode_lengths),
             np.arange(len(stats.episode_lengths)))
    plt.xlabel("Time Steps")
    plt.ylabel("Episode")
    plt.title("Episode per time step")
    if noshow:
        plt.close(fig3)
    else:
        plt.show(fig3)

    return fig1, fig2, fig3


def plot_subfigures(img_list, max_ncol=5, figsize=(9, 6)):

    n_img = len(img_list)
    max_list = (np.array(img_list)).max()
    min_list = (np.array(img_list)).min()

    norm = colors.Normalize(vmin=max_list, vmax=min_list)
    mappable = ScalarMappable(cmap='jet', norm=norm)
    mappable._A = []
    if n_img <= max_ncol:
        nrow = 1
    else:
        nrow = n_img // max_ncol
    fig, axs = plt.subplots(nrows=nrow, ncols=max_ncol,
                            figsize=figsize, sharex=True, sharey=True)
    for (ax, img) in zip(axs, img_list):
        ax.imshow(img, cmap='jet', vmin=min_list, vmax=max_list)
        # max_img, min_img = img.max(), img.min()
        # max_b = int(255*max_img/max_list)
        # ax.imshow(adjust_grayscale(img,max_b), cmap='jet')

    axpos = axs[-1].get_position()
    cbar_ax = fig.add_axes([axpos.x1, axpos.y1, 0.02, axpos.height])
    fig.colorbar(mappable, cax=cbar_ax)
    return fig


def plot_skeleton(node, ax, rgb):
    color = rgb2hex(rgb)
    n_node = node.shape[0]
    for i in range(n_node):
        if i + 1 < n_node:
            x1, y1, z1 = node[i]
            x2, y2, z2 = node[i+1]
            ax.plot([x1, x2], [y1, y2], [z1, z2],
                    "o-", color=color, ms=4, mew=0.5)


def get_cmap(n, name='jet'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)


def rgba2rgb(rgba):
    """(r,g,b,a) ===> (r,g,b)
    Arguments:
        rgba {tuple} -- (r,g,b,a)
    """
    (r, g, b, a) = rgba
    # r_ = r * a + (1.0 - a) * 255
    # g_ = g * a + (1.0 - a) * 255
    # b_ = b * a + (1.0 - a) * 255
    r_ = int(r * a * 255)
    g_ = int(g * a * 255)
    b_ = int(b * a * 255)
    return(r_, g_, b_)


def rgb2hex(rgb):
    """(r,g,b) ===> hex

    Arguments:
        rgb {tuple} -- (r,g,b))

    Returns:
        string -- i.e., '#008040'
    """
    (r, g, b) = rgb
    return '#%02x%02x%02x' % (r, g, b)

# ============================ TRANSFORM UTILS ============================


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


def rpy2mat(Re, seq='xyz'):
    r = R.from_euler(seq, Re, degrees=True)
    return r.as_matrix()


def simrpy2quat(Re):
    r = R.from_euler('yxz', Re, degrees=True)
    return r.as_quat()


def simrpy2mat(Re):
    r = R.from_euler('yxz', Re, degrees=True)
    return r.as_matrix()


def rotate_3d(p, R, origin=(0, 0, 0)):
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def calc_2points_distance(p1, p2):
    p1 = np.array(p1)
    p2 = np.array(p2)
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


def calc_lineseg_dist(p, l):
    """Function calculates the distance from point p to line segment [a,b].
    This function cannot calculate the shortest/perpendicular distance!!
    Use `calc_shortest_dist(p,l)`
    Arguments:
        p {list} -- [x,y,z] a point
        l {list} -- [p1.x,p1.y.p1.z, p2.x,p2.y.p2.z] a line

    Returns:
        {float} -- distance between p and l
    """
    p, l = np.array(p), np.array(l)
    a, b = l[0:3], l[3:6]

    a_ = np.array([a[0], 0, a[2]])
    b_ = np.array([b[0], 0, b[2]])

    da = a[1]
    db = b[1]

    ratio = np.linalg.norm(p - a_)/np.linalg.norm(b_ - a_)
    d = da + ratio*(db-da)
    return(d)

def calc_shortest_dist(p, l):
    """Function calculates the distance from point p to line segment [a,b].
    Parameters:
        p {list} -- [x,y,z] a point
        l {list} -- [p1.x,p1.y.p1.z, p2.x,p2.y.p2.z] a line
    Returns:
        {float} -- distance between p and l
    """
    p, l = np.array(p), np.array(l)
    a, b = l[0:3], l[3:6]

    # normalized tangent vector
    d = np.divide(b - a, np.linalg.norm(b - a))

    # signed parallel distance components
    s = np.dot(a - p, d)
    t = np.dot(p - b, d)

    # clamped parallel distance
    h = np.maximum.reduce([s, t, 0])

    # perpendicular distance component
    c = np.cross(p - a, d)
    return np.hypot(h, np.linalg.norm(c))

def is_between(a1, a2, b):
    """
    a1: [x, y] an end point on line segnment
    a2: [x, y] an end point on line segnment
    b : [x, y] a point
    """
    a1 = np.asarray(a1)
    a2 = np.asarray(a2)
    b = np.asarray(b)
    if abs(np.cross(a2-a1, b-a1)) > sys.float_info.epsilon: 
        return False
    if np.dot(a2-a1, b-a1) < 0:
        return False
    if np.dot(a2-a1, b-a1) > np.dot(a2-a1, a2-a1):
        return False
    return True
    

def calc_intersection(a1, a2, b1, b2):
    """
    Returns the point of intersection of the lines passing through a2,a1 and b2,b1.
    https://stackoverflow.com/questions/3252194/numpy-and-line-intersections
    a1: [x, y] a point on the first line
    a2: [x, y] another point on the first line
    b1: [x, y] a point on the second line
    b2: [x, y] another point on the second line
    """
    s = np.vstack([a1, a2, b1, b2])        # s for stacked
    h = np.hstack((s, np.ones((4, 1))))  # h for homogeneous
    l1 = np.cross(h[0], h[1])           # get first line
    l2 = np.cross(h[2], h[3])           # get second line
    x, y, z = np.cross(l1, l2)          # point of intersection
    if z == 0:                          # lines are parallel
        return (float('inf'), float('inf'))
    return [x/z, y/z]


def replace_bad_point(img, loc, bounding_size=20):
    """
    find the closest meaningful point for top-ranked grasp point
    to avoid some cases where the pixel value is zero or very low
    if the pixel value is 0 or 255, replace
    else return input values
    """

    (x, y) = loc
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY) if len(img.shape) > 2 else img
    if bounding_size > int(gray.shape[0]/2) or bounding_size > int(gray.shape[1]/2):
        bounding_size = min(int(gray.shape[0]/2), int(gray.shape[1]/2))
    
    # background_pixel = 10
    # if gray[y,x] < background_pixel:
    #     h, w = gray.shape
    #     mat = gray[(y-bounding_size):(y+bounding_size+1),
    #             (x-bounding_size):(x+bounding_size+1)]
    #     max_xy = np.where(mat == mat.max())
    #     # y_p = max_xy[0][0] + y-bounding_size
    #     # x_p = max_xy[1][0] + x-bounding_size
    #     x_p = max_xy[0][0] + y-bounding_size
    #     y_p = max_xy[1][0] + x-bounding_size

    #     return (x_p, y_p)
    # else:
    #     return loc
    h, w = gray.shape
    # mat = gray[(y-bounding_size):(y+bounding_size+1),
    #            (x-bounding_size):(x+bounding_size+1)]
    mat = gray[(y-bounding_size):(y+bounding_size),
               (x-bounding_size):(x+bounding_size)]
    # plt.imshow(mat), plt.show()
    max_xy = np.where(mat == mat.max())
    # y_p = max_xy[0][0] + y-bounding_size
    # x_p = max_xy[1][0] + x-bounding_size
    y_p = max_xy[0][0] + y-bounding_size
    x_p = max_xy[1][0] + x-bounding_size
    if gray[y_p, x_p] == gray[y,x]: return (x, y)
    # print(f"({x},{y}) ==> ({x_p},{y_p})")

    return (x_p, y_p)

def detect_ar_marker(image, marker_type="DICT_5X5_100", show=True):
    
    ARUCO_DICT = {
        "DICT_4X4_100": cv2.aruco.DICT_4X4_100,
        "DICT_5X5_100": cv2.aruco.DICT_5X5_100,
        "DICT_6X6_100": cv2.aruco.DICT_6X6_100}
    arucoDict = cv2.aruco.Dictionary_get(ARUCO_DICT[marker_type])
    arucoParams = cv2.aruco.DetectorParameters_create()
    (corners, ids, rejected) = cv2.aruco.detectMarkers(
           image, arucoDict, parameters=arucoParams)
    res = {}
    if len(corners) > 0:
        ids = ids.flatten()
        for (markerCorner, markerID) in zip(corners, ids):
            corners = markerCorner.reshape((4, 2))
            (topLeft, topRight, bottomRight, bottomLeft) = corners

            topRight = (int(topRight[0]), int(topRight[1]))
            bottomRight = (int(bottomRight[0]), int(bottomRight[1]))
            bottomLeft = (int(bottomLeft[0]), int(bottomLeft[1]))
            topLeft = (int(topLeft[0]), int(topLeft[1]))

            cv2.line(image, topLeft, topRight, (0, 255, 0), 2)
            cv2.line(image, topRight, bottomRight, (0, 255, 0), 2)
            cv2.line(image, bottomRight, bottomLeft, (0, 255, 0), 2)
            cv2.line(image, bottomLeft, topLeft, (0, 255, 0), 2)

            cX = int((topLeft[0] + bottomRight[0]) / 2.0)
            cY = int((topLeft[1] + bottomRight[1]) / 2.0)

            res[markerID] = [cX, cY]

            cv2.circle(image, (cX, cY), 4, (0, 0, 255), -1)
                # draw the ArUco marker ID on the image
            cv2.putText(image, str(markerID),
                        (cX, cY - 15), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 0, 255), 3)
        if show: 
            cv2.namedWindow("markers", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("markers", 1920, 1080)
            cv2.imshow("markers", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    return res 


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
        print("Error: x is too large! ")
    if(0.67 <= robot_y):
        print("Error: y is too large! ")
    # if(0.005152941176470586 >= robot_z):
    if(0.005 >= robot_z):
        print("Error: z is too small! ")
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
        [0,              1, 0],
        [-math.sin(rad), 0, math.cos(rad)]
    ])
    return np.dot(H, pc.T).T


def print_quaternion(q):
    q = np.round(q, 3)
    print(f"w: {q[3]}, x: {q[0]}, y: {q[1]}, z: {q[2]}")


def rotate_in_sim(p, xz, y):
    """first rotate along positive Y-axis, then minus xz-axis"""
    y_rot_p = (np.dot(rpy2mat([0, y, 0]), p.T)).T
    xz_rot_p = (np.dot(rpy2mat([-xz, 0, 0]), y_rot_p.T)).T
    return xz_rot_p

# ========================== VISION UTILS ===========================


def rotate_img(img, angle, center=None, scale=1.0, cropped=True):
    """
    angle: degree (countclockwise)
    """
    (h, w) = img.shape[:2]

    if center is None:
        center = (w/2, h/2)

    M = cv2.getRotationMatrix2D(center, angle, scale)

    if cropped is True:
        rotated = cv2.warpAffine(img, M, (w, h))
        return rotated

    else:
        cosM = np.abs(M[0][0])
        sinM = np.abs(M[0][1])

        new_h = int((h * sinM) + (w * cosM))
        new_w = int((h * cosM) + (w * sinM))

        M[0][2] += (new_w/2) - center[0]
        M[1][2] += (new_h/2) - center[1]

        # Now, we will perform actual image rotation
        rotated = cv2.warpAffine(img, M, (new_w, new_h))
        return cv2.resize(rotated, (h, w))


def rotate_img_kpt(img, kpts, angle, center=None, scale=1.0, cropped=True):
    """
    angle: degree (countclockwise)
    locs: np.array([[x1,y1],[x2,y2],...])
    return: rotated image, rotated keypoints np.array([[x1,y1],[x2,y2],...])
    """
    (h, w) = img.shape[:2]
    num_kpt = kpts.shape[0]

    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)
    input_kpts = np.c_[kpts, np.ones(num_kpt)]
    if cropped is True:
        rotated = cv2.warpAffine(img, M, (w, h))
        rot_kpts = np.dot(M, input_kpts.T)
        return rotated, rot_kpts.T[:, 0:2]
    else:
        cosM = np.abs(M[0][0])
        sinM = np.abs(M[0][1])

        new_h = int((h * sinM) + (w * cosM))
        new_w = int((h * cosM) + (w * sinM))
        M[0][2] += (new_w/2) - center[0]
        M[1][2] += (new_h/2) - center[1]

        # Now, we will perform actual image rotation
        rotated = cv2.warpAffine(img, M, (new_w, new_h))
        rot_kpts = np.dot(M, input_kpts.T).T[:, 0:2]
        rot_kpts[:, 0] *= (w/new_w)
        rot_kpts[:, 1] *= (h/new_h)
        return cv2.resize(rotated, (h, w)), rot_kpts


def rotate_3d(p, R, origin=(0, 0, 0)):
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)


def rotate_pixel(h, w, loc, angle, center=None, scale=1.0, cropped=True):
    x, y, = loc

    if center is None:
        center = (w/2, h/2)
    M = cv2.getRotationMatrix2D(center, angle, scale)

    if cropped == True:
        rot_loc = np.dot(M, [x, y, 1])
        return rot_loc
    else:
        cosM = np.abs(M[0][0])
        sinM = np.abs(M[0][1])

        new_h = int((h * sinM) + (w * cosM))
        new_w = int((h * cosM) + (w * sinM))

        M[0][2] += (new_w/2) - center[0]
        M[1][2] += (new_h/2) - center[1]

        rot_loc = np.dot(M, [x, y, 1])

        return rot_loc*(h/new_h)


def rotate_point(loc, angle, center=None, scale=1.0):
    rad = np.radians(angle)
    c, s = np.cos(rad), np.sin(rad)
    M = np.array(((c, -s), (s, c)))

    # M = cv2.getRotationMatrix2D(center, angle, scale)
    x, y = loc
    rot_loc = np.dot(M, [x, y])

    return rot_loc


def get_neighbor_bounding(gray, loc, bounding_size=10, bounding_draw=False):
    """Give a mat and a location, return the max neighobor pixel value
       Xinyi

    Arguments:
        gray {grayscale or array} -- images with 1 channels, shape = (h,w)
        loc {(int, int)} -- location, (left,top), (from width,from height) 

    Keyword Arguments:
        bounding_size {int} -- [neighbor bounding size] (default: {10})
        bounding_draw {bool} -- [if draw rectangle to verify position] (default: {False})

    Returns:
        [int] -- max value of neighbor pixels
    """
    (x, y) = loc
    h, w = gray.shape

    # construct the refined matrix, first row then column
    r_con = np.zeros((bounding_size, w), dtype=np.uint8)
    mat_tmp = np.r_[np.r_[r_con, gray], r_con]
    c_con = np.zeros((h+bounding_size*2, bounding_size), dtype=np.uint8)
    mat_tmp = np.c_[np.c_[c_con, mat_tmp], c_con]

    if bounding_draw:
        # draw the bounding box to verify the correct pixel
        # copy the image for draw
        for_draw = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB)
        cv2.circle(for_draw, (x, y), 6, (255, 0, 0), -1)
        cv2.rectangle(for_draw, ((x-bounding_size), (y-bounding_size)), ((x+bounding_size+1), (y+bounding_size+1)), (255, 0,0),2)
        plt.figure(figsize=(15, 15))
        plt.imshow(for_draw, cmap='gray'), plt.show()

    x = x+bounding_size
    y = y+bounding_size
    return mat_tmp[(y-bounding_size):(y+bounding_size+1), (x-bounding_size):(x+bounding_size+1)]


def get_max_neighbor_pixel(gray, loc, bounding_size=10, bounding_draw=False):
    """Give a mat and a location, return the max neighobor pixel value
       Xinyi

    Arguments:
        gray {grayscale or array} -- images with 1 channels, shape = (h,w)
        loc {(int, int)} -- location, (left,top), (from width,from height) 

    Keyword Arguments:
        bounding_size {int} -- [neighbor bounding size] (default: {10})
        bounding_draw {bool} -- [if draw rectangle to verify position] (default: {False})

    Returns:
        [int] -- max value of neighbor pixels
    """
    (x, y) = loc
    h, w = gray.shape

    # construct the refined matrix, first row then column
    r_con = np.zeros((bounding_size, w), dtype=np.uint8)
    mat_tmp = np.r_[np.r_[r_con, gray], r_con]
    c_con = np.zeros((h+bounding_size*2, bounding_size), dtype=np.uint8)
    mat_tmp = np.c_[np.c_[c_con, mat_tmp], c_con]

    if bounding_draw:
        # draw the bounding box to verify the correct pixel
        # copy the image for draw
        for_draw = cv2.cvtColor(gray.copy(), cv2.COLOR_GRAY2RGB)
        cv2.circle(for_draw, (x, y), 6, (255, 0, 0), -1)
        cv2.rectangle(for_draw, ((x-bounding_size), (y-bounding_size)), ((x+bounding_size+1), (y+bounding_size+1)), (255, 0,0),2)
        plt.figure(figsize=(15, 15))
        plt.imshow(for_draw, cmap='gray'), plt.show()

    x = x+bounding_size
    y = y+bounding_size
    return np.max(mat_tmp[(y-bounding_size):(y+bounding_size+1), (x-bounding_size):(x+bounding_size+1)])


def get_neighbor_pixel(gray, loc, bounding_size=10):
    (x, y) = loc
    h, w = gray.shape
    mat = gray[(y-bounding_size):(y+bounding_size+1),
               (x-bounding_size):(x+bounding_size+1)]
    left_margin = x-bounding_size
    top_margin = y-bounding_size

    # index = w * Y + X
    max_xy = np.where(mat == mat.max())

    y_p = max_xy[0][0] + y-bounding_size
    x_p = max_xy[1][0] + x-bounding_size
    return (x_p, y_p), mat.max()


def normalize_depth_map(gray_array, max_distance, min_distance, img_width, img_height):
    """normalize an array to range[0,255], display(optional)

    Arguments:
        gray_array {numpy array} -- 1channel, with camera z-value, unit: mm
        max_distance {int} -- camera max distance
        min_distance {int} -- camera min distance
        img_width {int} -- depth image width, need to be the same as camera
        img_height {int} -- depth image height, need to be the same as camera

    Returns:
        grayscale image -- depth map
    """

    gray_array[gray_array > max_distance] = max_distance
    gray_array[gray_array < min_distance] = min_distance
    # gray_array = - gray_array
    img = ((gray_array - min_distance) *
           (1/(max_distance - min_distance) * 255)).astype('uint8')
    img = (-img).reshape((img_height, img_width))

    # img[img>=225]=80
    # remove pixels of 255

    # cv2.imshow("windows", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img


def adjust_grayscale(img, max_b=255):

    # adjust depth maps
    img_adjusted = np.array(img * (max_b/np.max(img)), dtype=np.uint8)
    return img_adjusted


def adjust_array_range(array, range=(0, 1), if_img=False):
    # adjust depth maps
    if if_img == False:
        return np.interp(array, (array.min(), array.max()), range)
    else:
        return np.array(np.interp(array, (array.min(), array.max()), range), dtype=np.uint8)


def depth2pc(gray, pc_path=None):
    import open3d as o3d
    h, w = gray.shape
    y_ = np.linspace(1, h, h)
    x_ = np.linspace(1, w, w)
    mesh_x, mesh_y = np.meshgrid(x_, y_)
    z_ = gray.flatten()

    xyz = np.zeros((np.size(mesh_x), 3))

    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_, -1)
    xyz = np.delete(xyz, np.where(xyz[:, 2] == 0)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)
    o3d.visualization.draw_geometries([pcd])

    if pc_path is not None:
        o3d.io.write_point_cloud(pc_path, pcd)

    # o3d.io.read_point_cloud(pc_path)
    # o3d.visualization.draw_geometries([pcd])
    return pcd


def depth2xyz(gray):
    h, w = gray.shape
    y_ = np.linspace(1, h, h)
    x_ = np.linspace(1, w, w)
    mesh_x, mesh_y = np.meshgrid(x_, y_)
    z_ = gray.flatten()

    xyz = np.zeros((np.size(mesh_x), 3))

    xyz[:, 0] = np.reshape(mesh_x, -1)
    xyz[:, 1] = np.reshape(mesh_y, -1)
    xyz[:, 2] = np.reshape(z_, -1)
    xyz = np.delete(xyz, np.where(xyz[:, 2] == 0)[0], axis=0)
    return xyz


def xyz2depth(gray_array, max_distance, min_distance, img_width, img_height):
    """normalize an array to range[0,255], display(optional)

    Arguments:
        gray_array {numpy array} -- 1channel, with camera z-value, unit: mm
        max_distance {int} -- camera max distance
        min_distance {int} -- camera min distance
        img_width {int} -- depth image width, need to be the same as camera
        img_height {int} -- depth image height, need to be the same as camera

    Returns:
        grayscale image -- depth map
    """

    gray_array[gray_array > max_distance] = max_distance
    gray_array[gray_array < min_distance] = min_distance
    # gray_array = - gray_array
    img = ((gray_array - min_distance) *
           (1 / (max_distance - min_distance) * 255)).astype('uint8')
    img = (-img).reshape((img_height, img_width))

    img_adjusted = np.array(img * (255/np.max(img)), dtype=np.uint8)

    # remove pixels of 255

    # cv2.imshow("windows", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    return img_adjusted


def process_raw_pc(pcd_path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(pcd_path)

    # o3d.visualization.draw_geometries([pcd])

    xyz = np.asarray(pcd.points)
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    # pc_path = "./vision/tmp/reform.ply"
    # o3d.io.write_point_cloud(pc_path, pcd)
    # re_pcd = o3d.io.read_point_cloud(pc_path)
    # down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    # re_xyz = np.asarray(down_pcd.points)
    # return (re_xyz)
    # pc_path = "./vision/tmp/reform.ply"
    # o3d.io.write_point_cloud(pc_path, pcd)
    # re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)


def process_raw_xyz(xyz_path):
    import open3d as o3d
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
    import open3d as o3d
    xyz = np.delete(xyz, np.where(xyz[:, 1] <= 1)[0], axis=0)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(xyz)

    pc_path = "./vision/tmp/reform.ply"
    o3d.io.write_point_cloud(pc_path, pcd)
    re_pcd = o3d.io.read_point_cloud(pc_path)
    down_pcd = re_pcd.voxel_down_sample(voxel_size=8)

    re_xyz = np.asarray(down_pcd.points)
    return (re_xyz)


def rigid_transform_3D(A, B):
    assert A.shape == B.shape

    num_rows, num_cols = A.shape
    if num_rows != 3:
        raise Exception(f"matrix A is not 3xN, it is {num_rows}x{num_cols}")

    num_rows, num_cols = B.shape
    if num_rows != 3:
        raise Exception(f"matrix B is not 3xN, it is {num_rows}x{num_cols}")

    # find mean column wise
    centroid_A = np.mean(A, axis=1)
    centroid_B = np.mean(B, axis=1)

    # ensure centroids are 3x1
    centroid_A = centroid_A.reshape(-1, 1)
    centroid_B = centroid_B.reshape(-1, 1)

    # subtract mean
    Am = A - centroid_A
    Bm = B - centroid_B

    H = Am @ np.transpose(Bm)

    # sanity check
    # if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = -R @ centroid_A + centroid_B

    return R, t


def validate_transform(A, R, t):
    #A = calib_camera
    B = R@A + t

    # Recover R and t
    ret_R, ret_t = rigid_transform_3D(A, B)

    # Compare the recovered R and t with the original
    B2 = (ret_R@A) + ret_t

    n = 9

    # Find the root mean squared error
    err = B2 - B
    err = err * err
    err = np.sum(err)
    rmse = np.sqrt(err/n)

    print("Points A")
    print(A)

    print("Points B")
    print(B)

    print("Ground truth rotation")
    print(R)

    print("Recovered rotation")
    print(ret_R)

    print("Ground truth translation")
    print(t)

    print("Recovered translation")
    print(ret_t)

    print("RMSE:", rmse)

    if rmse < 1e-5:
        print("Everything looks good!")
    else:
        print("Hmm something doesn't look right ...")

# img = cv2.imread("C:\\Users\\xinyi\\Downloads\\twice.jpg")
# img = cv2.resize(img, (500,500))
# kpt = np.array([[217,197],[325,191],[271,119]])
# for p in kpt:
#     cv2.circle(img, p, 5, (0,0,255), -1)
#     print(p)

# rot_img, rot_kpt = rotate_img_kpt(img, kpt, 45, cropped=False)

# for p in rot_kpt:
#     print(p)
#     cv2.circle(rot_img, (int(p[0]),int(p[1])), 5, (0,0,255), -1)
# cv2.imshow("windows", rot_img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
# # plt.imshow(rot_img)
# # plt.show()
