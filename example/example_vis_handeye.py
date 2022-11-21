import numpy as np
import open3d as o3d
from bpbot.utils import *
import copy

coor = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
# import bpbot.driver.phoxi.phoxi_client as pclt
# pxc = pclt.PhxClient(host="127.0.0.1:18300")
# pxc.triggerframe()
# pxc.saveply("/tmp/test_ply.ply")
# gray = pxc.getgrayscaleimg()
# image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
# detect_ar_marker(image)

                                
# ply_point_cloud = o3d.data.PLYPointCloud()
# pcd = o3d.io.read_point_cloud(ply_point_cloud.path)

# with o3d.utility.VerbosityContextManager(
#         o3d.utility.VerbosityLevel.Debug) as cm:
#     labels = np.array(
#         pcd.cluster_dbscan(eps=0.02, min_points=10, print_progress=True))

# max_label = labels.max()
# print(f"point cloud has {max_label + 1} clusters")
# colors = plt.get_cmap("tab20")(labels / (max_label if max_label > 0 else 1))
# colors[labels < 0] = 0
# print(colors.shape)
# print(colors[:,:3].dtype)
# pcd.colors = o3d.utility.Vector3dVector(colors[:, :3])
# o3d.visualization.draw_geometries([pcd],
#                                   zoom=0.455,
#                                   front=[-0.4999, -0.1659, -0.8499],
#                                   lookat=[2.1813, 2.0619, 2.0999],
#                                   up=[0.1204, -0.9852, 0.1215])
# pcd_path = "/home/hlab/choreonoid-1.7.0/ext/graspPlugin/PCL/test_ply.ply"
# idx = 4
# idx_pc = idx + 1
calib_dir = os.path.join("/home/hlab/bpbot", "data/calibration", "test")
save_robot = os.path.join(calib_dir, "robot_clb.txt")
save_cam = os.path.join(calib_dir, "camera_clb.txt")
G = np.loadtxt(os.path.join(calib_dir, "calibmat.txt"))

# p_cam = np.vstack([np.loadtxt(save_cam_l), np.loadtxt(save_cam_r)])
# p_robot = np.vstack([np.loadtxt(save_robot_l), np.loadtxt(save_robot_r)])
p_cam = np.loadtxt(save_cam)
p_robot = np.loadtxt(save_robot)
p_cam /= 1000
p_robot[:,0] += 0.079
p_robot[:,2] -= 0.030

for idx in range(40):

    pcd_path = f"/home/hlab/Desktop/pcd/{1+idx:02d}.ply"
    # ------------------ cam coordinate ---------------------
    # pcd_cam = o3d.geometry.PointCloud()
    # pcd_cam.points = o3d.utility.Vector3dVector(p_cam)
    # rgb = np.tile([0.5,0,0], (np.asarray(pcd_cam.points).shape[0],1))
    # pcd_cam.colors = o3d.utility.Vector3dVector(rgb)

    pcd = o3d.io.read_point_cloud(pcd_path)
    # rgb = np.tile([0.8,0.8,0.8], (np.asarray(pcd.points).shape[0],1))
    # pcd.colors = o3d.utility.Vector3dVector(rgb)

    # mesh = o3d.geometry.TriangleMesh.create_coordinate_frame()
    # mesh.scale(0.25, center=mesh.get_center())
    # o3d.visualization.draw_geometries([mesh, pcd, pcd_cam])

    # ------------------ robot coordinate ---------------------
    pcd_robot = o3d.geometry.PointCloud()
    pcd_robot.points = o3d.utility.Vector3dVector(p_robot)
    rgb = np.tile([0.6,0,0], (p_robot.shape[0],1))
    rgb[idx] = np.array([1,0,0])
    pcd_robot.colors = o3d.utility.Vector3dVector(rgb)
    # print("original: ", p_robot[idx])
    coor_pr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.09, origin=p_robot[idx])

    # transformation #1 collected point
    _p_cam = np.c_[p_cam, np.ones(p_cam.shape[0])]
    p_cam2robot = np.dot(G, _p_cam.T).T[:,:3]
    pcd_cam2robot = o3d.geometry.PointCloud()
    pcd_cam2robot.points = o3d.utility.Vector3dVector(p_cam2robot)
    rgb = np.tile([0,0.6,0], (p_cam2robot.shape[0],1))
    rgb[idx] = np.array([0,1,0])
    pcd_cam2robot.colors = o3d.utility.Vector3dVector(rgb)
    coor_pc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.09, origin=p_cam2robot[idx])
    # print("converted: ", p_cam2robot[idx])

    # tranformation #2 point cloud 
    p = np.asarray(pcd.points)
    _p = np.c_[p, np.ones(p.shape[0])]
    p_cvt = np.dot(G, _p.T).T[:,:3]
    p_cvt_down = np.delete(p_cvt, np.where(p_cvt[:,2] < -0.05)[0], axis=0)
    pcd_cvt = o3d.geometry.PointCloud()
    pcd_cvt.points = o3d.utility.Vector3dVector(p_cvt_down)
    rgb = np.tile([0.686,0.933,0.933], (p_cvt_down.shape[0],1))
    pcd_cvt.colors = o3d.utility.Vector3dVector(rgb)

    o3d.visualization.draw_geometries([coor, pcd_cvt, pcd_robot, pcd_cam2robot, coor_pc])