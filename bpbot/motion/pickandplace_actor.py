import math
from bpbot import BinConfig

class PickAndPlaceActor(object):
    def __init__(self, filepath):
        
        cfg = BinConfig()
        cfgdata = cfg.data
        if filepath is None: 
            self.filepath = cfg.motionfile_path
        else:
            self.filepath = filepath 
        w_lft = (cfgdata["hand"]["left"]["open_width"]/2/1000) * 180 / math.pi
        w_rgt = (cfgdata["hand"]["right"]["open_width"]/2/1000) * 180 / math.pi
        
        self.initpose = "0 1.00 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 %.3f %.3f %.3f %.3f" % (w_rgt,-w_rgt,w_lft,-w_lft)
        self.bothhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f %.3f %.3f"% (w_rgt,-w_rgt,w_lft,-w_lft) 
        self.lhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f"% (w_lft,-w_lft) 
        self.rhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f 0 0"% (w_rgt,-w_rgt) 
       
        self.front_c = [0.440, 0.310]
        self.side_c = [0.070, 0.552]
        self.drop_bin = [0.418, 0.250]
    
    def gen_empty_action(self):
        open(self.filepath, 'w').close()
    
    def get_pick_seq(self, xyz, rpy):
        return [
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 0.80 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
            "0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",
            "0 0.80 LARM_XYZ_ABS %.3f %.3f 0.320 %.1f %.1f %.1f" % (*xyz[:2], *rpy)
        ]

    def get_place_seq(self, rpy, dest):
        if dest == "front":
            return [
                "0 0.80 LARM_XYZ_ABS %.3f %.3f 0.320 %.1f %.1f %.1f" % (*self.front_c,*rpy),
                "0 0.50 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.front_c,*rpy),
                "0 0.50 LHAND_JNT_OPEN",
                self.initpose
            ]
        elif dest == "side":
            return [
                # "0 1.50 LARM_XYZ_ABS %.3f %.3f 0.350 -90.0 -90.0 90.0" % (*self.goal_c,),
                # "0 0.50 LARM_XYZ_ABS %.3f %.3f 0.200 -90.0 -90.0 90.0" % (*self.goal_c,),
                "0 1.50 LARM_XYZ_ABS %.3f %.3f 0.320 %.1f %.1f %.1f" % (*self.side_c,*rpy),
                "0 0.50 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*self.side_c,*rpy),
                "0 0.50 LHAND_JNT_OPEN",
                self.initpose
            ]
     
    def get_action(self, pose, dest="front"):
        """Get action for pick-and-place motion

        Args:
            pose (tuple or list): grasp pose
            dest (str, optional): destination, "front", "side", "drop". Defaults to "front".
        """
        xyz = pose[:3]
        rpy = pose[3:]
        if dest == "drop":
            seqs = self.get_pick_seq(xyz, rpy) + self.get_drop_seq(rpy)
        else:
            seqs = self.get_pick_seq(xyz, rpy) + self.get_place_seq(rpy, dest=dest)
        with open(self.filepath, 'wt') as fp:
            for s in seqs:
                print(s, file=fp)

    def get_drop_seq(self, rpy):
        return [
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.350 %.1f %.1f %.1f" % (*self.drop_bin,*rpy),
            "0 0.80 LARM_XYZ_ABS %.3f %.3f 0.300 %.1f %.1f %.1f" % (*self.drop_bin,*rpy),
            "0 0.50 LHAND_JNT_OPEN",
            self.initpose
        ]