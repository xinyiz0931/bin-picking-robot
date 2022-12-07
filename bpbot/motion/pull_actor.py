import numpy as np
import math
from bpbot import BinConfig

class PullActor(object):
    def __init__(self, filepath):
        cfg = BinConfig()
        cfgdata = cfg.data
        if filepath is None: 
            self.filepath = cfg.motionfile_path
        else:
            self.filepath = filepath 
        w_lft = (cfgdata["hand"]["left"]["open_width"]/2/1000) * 180 / math.pi
        w_rgt = (cfgdata["hand"]["right"]["open_width"]/2/1000) * 180 / math.pi 

        self.initpose = "0 0.80 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 %.3f %.3f %.3f %.3f" % (w_rgt,-w_rgt,w_lft,-w_lft)
        self.bothhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f %.3f %.3f"% (w_rgt,-w_rgt,w_lft,-w_lft) 
        self.lhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f"% (w_lft,-w_lft) 
        self.rhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f 0 0"% (w_rgt,-w_rgt) 
       
        self.side_c = [0.070, 0.552]
        self.drop_c = [0.438, 0.200]

    def get_pick_seq(self, xyz, rpy): 
        return [
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 1.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
            "0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0"
        ]
    
    def get_place_seq(self, rpy, dest="side"):
        if dest == "front":
            return [
                "0 0.80 LARM_XYZ_ABS %.3f %.3f 0.300 %.1f %.1f %.1f" % (*self.drop_c, *rpy),
                # "0 0.50 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.drop_c, *rpy),
                "0 0.50 LHAND_JNT_OPEN",
                self.initpose
            ]
        elif dest == "side":
            return [
                "0 1.50 LARM_XYZ_ABS %.3f %.3f 0.350 %.1f %.1f %.1f" % (*self.side_c, *rpy),
                "0 0.50 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*self.side_c, *rpy),
                "0 0.50 LHAND_JNT_OPEN",
                self.initpose
            ]

    def get_action(self, pose, v, wiggle=False):
        """Get pulling motionfile command

        Args:
            xyz (tuple or list): [x,y,z]
            rpy (tuple or list): [r,p,y]
            v (tuple or list): [x,y,z,length]
            wiggle (bool, optional): pulling with wiggling?. defaults to False.
        """
        xyz = pose[:3]
        rpy = pose[3:]

        xyz_e = [xyz[i] + v[3]*v[i] for i in range(3)]
        xyz_u = xyz_e.copy()
        xyz_u[2] += 0.1 

        if wiggle:
            pull_seq = self.get_wiggle_seq(xyz, rpy, xyz_e, 8) + ["0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_u, *rpy)] 
        else:
            pull_seq = [
                "0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_e, *rpy),
                "0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_u, *rpy)
            ]  
        
        seqs = self.get_pick_seq(xyz, rpy) + pull_seq + self.get_place_seq(rpy)
        with open(self.filepath, 'wt') as fp:
            for s in seqs:
                print(s, file=fp)

    def get_wiggle_seq(self, xyz_s, rpy, xyz_e, itvl=8):
        seq = []
        rpy_bfr, rpy_aft = rpy.copy(), rpy.copy()
        rpy_bfr[1] += 3
        rpy_aft[1] -= 3
        for i in range(itvl):
            _xyz = [xyz_s[k]+(xyz_e[k]-xyz_s[k])/itvl*(i+1) for k in range(3)]
            seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*_xyz, *rpy_bfr))
            seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*_xyz, *rpy_aft))
        return seq
    
    def get_holdandpull_seq(self):
        # TODO
        return
