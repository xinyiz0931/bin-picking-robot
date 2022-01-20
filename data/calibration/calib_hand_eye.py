import os
import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import configparser
np.set_printoptions(suppress=True)

# Input: expects 3xN matrix of points
# Returns R,t
# R = 3x3 rotation matrix
# t = 3x1 column vector

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
    #if linalg.matrix_rank(H) < 3:
    #    raise ValueError("rank of H = {}, expecting 3".format(linalg.matrix_rank(H)))

    # find rotation
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # special reflection case
    if np.linalg.det(R) < 0:
        print("det(R) < R, reflection detected!, correcting for it ...")
        Vt[2,:] *= -1
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

if __name__ == "__main__":

    # ========================== define path =============================
    ROOT_DIR = os.path.abspath("./")
    config_path = os.path.join(ROOT_DIR, "config.ini")
    calib_path = os.path.join(ROOT_DIR, "vision/calibration/")

    # ======================= get config info ============================
    config = configparser.ConfigParser()
    config.read(config_path)
    
    width = int(config['GRASP']['width'])
    height = int(config['GRASP']['height'])

    # ======================== get point sets ============================
    pcd = o3d.io.read_point_cloud("../calibration/calib.ply") # camera coordinate (default!!!)
    
    img_point = np.loadtxt(calib_path+"camera.txt",delimiter=',')
    robot_point = np.loadtxt(calib_path+"robot.txt",delimiter=',')
    camera_point = np.asarray(pcd.points)

    # check the num of point sets
    if img_point.shape[0] == robot_point.shape[0]:
        calib_point_num = img_point.shape[0]
    else:
        print("Make sure the point sets have the same size..")

    calib_camera = np.array([])
    calib_image = np.array([])
    for p in img_point:
        u,v = int(p[0]),int(p[1])
        offset = width*v + u
        calib_image = np.append(calib_image, [u,v,camera_point[offset][2]])
        calib_camera = np.append(calib_camera, camera_point[offset])
    
    calib_image = calib_image.reshape((calib_point_num,3))
    # print(robot_point.T)
    calib_camera = (calib_camera/1000).reshape((calib_point_num,3))
    robot_point = robot_point.reshape((calib_point_num,3))

    # ======================= start calibration ==========================
    R ,t = rigid_transform_3D(calib_camera.T,robot_point.T)
    
    # assembly to 4x4 matrix and write in the calib_mat.txt
    H = np.r_[np.c_[R,t],[[0,0,0,1]]]
    print(H)
    # np.savetxt(calib_path+'calibmattt.txt',H)