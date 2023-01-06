import os
import re
import cv2
import argparse
import numpy as np
import importlib
FOUND_CNOID = importlib.util.find_spec("cnoid") is not None
if FOUND_CNOID:
    from cnoid.Util import *
    from cnoid.Base import *
    from cnoid.Body import *
    from cnoid.BodyPlugin import *
    from cnoid.GraspPlugin import *
    from cnoid.BinPicking import *

from bpbot.robotcon import NxtRobot
from bpbot.device import PhxClient
from bpbot.utils import *

def main():
    actor = HandEyeCalib("calib", "./data/calibration/test")
    parser = argparse.ArgumentParser('Hand eye calibration tool')
    parser.add_argument('-f', '--file', help='calibration result folder')
    parser.add_argument('-m', '--mode', choices=["calib", "calc", "vis"], required=True)

    args = parser.parse_args()
    print("[*] Start hand eye calibration in mode:", args.mode)
    if args.mode == "calib":
        actor.detect()
        actor.calc()
    elif args.mode == "calc":
        actor.calc()
    elif args.mode == "vis":
        actor.vis()

class HandEyeCalib(object):
    def __init__(self, mode, folder):
        self.mode = mode

        calib_dir = os.path.realpath(folder)
        self.save_hand = os.path.join(calib_dir, "pos_hand.txt")
        self.save_eye = os.path.join(calib_dir, "pos_eye.txt")
        self.save_mat = os.path.join(calib_dir, "calibmat.txt")

        mf_path = os.path.join(calib_dir, "calib_3d.dat")
        
    
        if self.mode == "calib" and FOUND_CNOID:
            self.robot = NxtRobot(host='[::]:15005')
            self.camera = PhxClient(host="127.0.0.1:18300")
            success = load_motionfile(mf_path, dual_arm=False)
            motion_seq = get_motion()
            num_seq = int(len(motion_seq)/20)
            self.motion_seq = np.reshape(motion_seq, (num_seq, 20))
            print(f"[*] Finish loading {num_seq} motion sequences! ")
            
        self.inipos_hand = self.extract_motionfile(mf_path)
    
    def detect(self, mkid=7):
        pos_eye, pos_hand = [], []
        if FOUND_CNOID: 
            for i, m in enumerate(self.motion_seq):
                
                self.robot.setJointAngles(m[1:],tm=m[0]) 

                self.camera.triggerframe()
                gray = self.camera.getgrayscaleimg()
                image = cv2.txColor(gray, cv2.COLOR_GRAY2RGB)
                pcd = self.camera.getpcd()
                ply_path = os.path.join("/home/hlab/Desktop/pcd/", f"{i:02d}.ply")
                self.camera.saveply(ply_path)

                ids = detect_ar_marker(image.copy(), show=False)
                
                if i == 0:
                    # intiial pose contains in the planner, therefore, skip the first motion
                    continue
                print(f"[*] {i:02d}-th | ", end="")

                if self.mkid in ids.keys(): 
                    x, y = ids[mkid]
                    # camera_p = pcd_r[y*image.shape[1]+x] / 1000
                    p = pcd[y*image.shape[1]+x]
                    pos_eye.append(p)
                    pos_hand.append(self.inipos_hand[i-1])
                    print(f"=> Detected #{mkid}, ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
                else: print(f"[!] No markers detected! ")
            pos_eye = np.asarray(pos_eye)
            pos_hand = np.asarray(pos_hand)

            # unit: m, transfer joint position to board position
            pos_eye /= 1000
            pos_hand[:,0] += 0.079
            pos_hand[:,2] -= 0.030 

            np.savetxt(np.asarray(self.save_eye), pos_eye, fmt='%.06f')
            np.savetxt(np.asarray(self.save_hand), pos_hand, fmt='%.06f')

            return pos_eye, pos_hand
        else:
            print("[!] Cannot detect ...")

    def calc(self, pos_eye=None, pos_hand=None):
        if pos_eye is None or pos_hand is None:
            pos_eye = np.loadtxt(self.save_eye)
            pos_hand = np.loadtxt(self.save_hand)

        if len(pos_eye) != len(pos_hand):
            print("[!] The numbers of positions are wrong ... ")    
            return
        
        R, t = rigid_transform_3D(pos_eye.T, pos_hand.T)

        print("-------------------------------")
        print(pos_eye.shape, pos_hand.shape)
        H = np.r_[np.c_[R, t], [[0, 0, 0, 1]]]
        print(H)
        print("-------------------------------")
        np.savetxt(self.save_mat, H, fmt='%.06f')
        return H

    def vis(self, pos_eye=None, pos_hand=None, pcd_path=None):
        import open3d as o3d
        if pos_eye is None or pos_hand is None:
            pos_eye = np.loadtxt(self.save_eye)
            pos_hand = np.loadtxt(self.save_hand)
        
        # prepare 
        G = np.loadtxt(self.save_mat)
        coor_zero = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.05, origin=[0, 0, 0])
        items = [coor_zero]

        # ------------------ robot coordinate ---------------------
        pcd_hand = o3d.geometry.PointCloud()
        pcd_hand.points = o3d.utility.Vector3dVector(pos_hand)
        rgb = np.tile([0.6,0,0], (pos_hand.shape[0],1))
        # rgb[idx] = np.array([1,0,0])
        pcd_hand.colors = o3d.utility.Vector3dVector(rgb)
        items.append(pcd_hand)
        # print("original: ", p_robot[idx])
        # coor_pr = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.09, origin=p_robot[idx])

        # transformation #1 collected point
        _pos_eye = np.c_[pos_eye, np.ones(pos_eye.shape[0])]
        pos_eye_tx = np.dot(G, _pos_eye.T).T[:,:3]
        pcd_eye_tx = o3d.geometry.PointCloud()
        pcd_eye_tx.points = o3d.utility.Vector3dVector(pos_eye_tx)
        rgb = np.tile([0,0.6,0], (pos_eye_tx.shape[0],1))
        # rgb[idx] = np.array([0,1,0])
        pcd_eye_tx.colors = o3d.utility.Vector3dVector(rgb)
        items.append(pcd_eye_tx)
        # coor_pc = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.09, origin=p_cam2robot[idx])
        # print("converted: ", p_cam2robot[idx])

        # tranformation #2 point cloud 
        if pcd_path is not None:
            pcd = o3d.io.read_point_cloud(pcd_path)
            p = np.asarray(pcd.points)
            _p = np.c_[p, np.ones(p.shape[0])]
            p_tx = np.dot(G, _p.T).T[:,:3]
            p_tx_down = np.delete(p_tx, np.where(p_tx[:,2] < -0.05)[0], axis=0)
            pcd_tx = o3d.geometry.PointCloud()
            pcd_tx.points = o3d.utility.Vector3dVector(p_tx_down)
            rgb = np.tile([0.686,0.933,0.933], (p_tx_down.shape[0],1))
            pcd_tx.colors = o3d.utility.Vector3dVector(rgb)
            items.append(pcd_tx)

        o3d.visualization.draw_geometries(items)
        
        return
    
    def extract_number(self, string):
        return re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", string)

    def extract_motionfile(self, filepath):
        """Extract robot pose (xyz+rpy) from motion file

        Args:
            filepath (str): path to motion file
        Returns:
            xyz (array): shape=(N,3), extracted poses
        """
        xyz = []
        with open(filepath, 'r+') as fp:
            lines = fp.readlines()
        for line in lines:
            if "#" in line: # comment in motion file 
                continue
            # print(line.split(' '))
            xyz.append([float(x) for x in self.extract_number(line)[2:5]])
        return np.asarray(xyz)

if __name__ == '__main__':
    main()