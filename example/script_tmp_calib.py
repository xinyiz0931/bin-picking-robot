import cv2
import numpy as np

from cnoid.Util import *
from cnoid.Base import *
from cnoid.Body import *
from cnoid.BodyPlugin import *
from cnoid.GraspPlugin import *
from cnoid.BinPicking import *



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

from datetime import datetime as dt
from bpbot.robotcon.nxt.nxtrobot_client import NxtRobot
import bpbot.driver.phoxi.phoxi_client as pclt
from bpbot.utils import *
pxc = pclt.PhxClient(host="127.0.0.1:18300")

calib_mk_id = 24

root_dir = "/home/hlab/bpbot"

tdatetime = dt.now()
tstr = tdatetime.strftime('%Y%m%d%H%M%S')
calib_dir = os.path.join(root_dir, "data/calibration")

mf_path = os.path.join(root_dir, "data/motion/calib_reverse.dat")

# robot_pos = []
# with open(mf_path, "r") as f:
#     for line in f.readlines():
#         robot_pos.append([float(i) for i in line.split(" ")[3:6]])
robot_pos = np.loadtxt(os.path.join(calib_dir, "20220714", "robot.txt"))
camera_pos = np.loadtxt(os.path.join(calib_dir, "20220714", "camera.txt"))
print(robot_pos.shape)
robot_pos[:,2] += (0.135+0.0017)
# plan_success = load_motionfile(mf_path)
# camera_pos = []

# if True:
#     nxt = NxtRobot(host='[::]:15005')
#     motion_seq = get_motion()
#     num_seq = int(len(motion_seq)/21)
#     print(f"[*] Total {num_seq} motion sequences! ")
#     motion_seq = np.reshape(motion_seq, (num_seq, 21))
#     for i, m in enumerate(motion_seq):
        
#         if m[1] == 0: 
#             nxt.closeHandToolLft()
#         elif m[1] == 1:
#             nxt.openHandToolLft()
#         nxt.setJointAngles(m[2:21],tm=m[0]) # no hand open-close control
    
#         print(f"[*] Start {i}-th capture! ")
#         pxc.triggerframe()
#         gray = pxc.getgrayscaleimg()
#         image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
#         clone = image.copy()

#         pcd = pxc.getpcd()
#         pcd_r = rotate_point_cloud(pcd)
#         id_locs = detect_ar_marker(image.copy(), show=False)
#         if id_locs != {}: 
#             print(f"[*] Detected marker: {id_locs}")
#             x, y = id_locs[calib_mk_id]
            
#             camera_p = pcd_r[y*image.shape[1]+x] / 1000
#             camera_pos.append(camera_p)
#         else: print(f"[*] No markers detected! ")

# camera_pos = np.asarray(camera_pos)
np.savetxt(os.path.join(calib_dir,"20220714", "camera.txt"), camera_pos)

print(camera_pos.shape, robot_pos.shape)
R, t = rigid_transform_3D(camera_pos.T, robot_pos.T)

# assembly to 4x4 matrix and write in the calib_mat.txt
H = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
print(H)

print(np.loadtxt(os.path.join(calib_dir, "old_result", "calibmat.txt")))
np.savetxt(os.path.join(calib_dir,"20220714", "calibmat.txt"), H)