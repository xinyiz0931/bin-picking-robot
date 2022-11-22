import math
from bpbot import BinConfig

class HelixActor(object):
    def __init__(self, filepath):
        self.filepath = filepath
        
        bincfg = BinConfig()
        cfg = bincfg.data
        w_lft = (cfg["hand"]["left"]["open_width"]/2/1000) * 180 / math.pi
        w_rgt = (cfg["hand"]["right"]["open_width"]/2/1000) * 180 / math.pi
        
        self.initpose = "0 1.00 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 %.3f %.3f %.3f %.3f" % (w_rgt,-w_rgt,w_lft,-w_lft)
        self.goal_c = [0.070, 0.552]
    
        self.half_helix = [
            "0 1.00 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90  145", # o
            "0 0.50 LARM_XYZ_ABS 0.600  0.120 0.340 -180 -90  145", # half
            "0 0.50 LARM_XYZ_ABS 0.627 -0.140 0.360 -180 -90  145", # half
            "0 0.75 LARM_XYZ_ABS 0.490 -0.180 0.400 -180 -90  145", # half
            "0 0.75 LARM_XYZ_ABS 0.537  0.260 0.420 -180 -90  145", # o
        ]
        self.helix = [
            "0 1.00 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90  145", # o
            "0 0.50 LARM_XYZ_ABS 0.600  0.120 0.340 -180 -90  145", # full
            "0 0.50 LARM_XYZ_ABS 0.627 -0.140 0.360 -180 -90  145", # full
            "0 0.50 LARM_XYZ_ABS 0.537 -0.320 0.380 -180 -90  145", # full
            "0 0.75 LARM_XYZ_ABS 0.490 -0.180 0.400 -180 -90  145", # full
            "0 0.75 LARM_XYZ_ABS 0.437  0.080 0.410 -180 -90  145", # full
            "0 0.75 LARM_XYZ_ABS 0.537  0.260 0.420 -180 -90  145", # o
            "0 0.50 LARM_XYZ_ABS 0.557  0.210 0.430  155 -73 -176", # two full
            "0 0.55 LARM_XYZ_ABS 0.580  0.010 0.440  160 -57  170", # two full
            "0 0.55 LARM_XYZ_ABS 0.534 -0.180 0.440  174 -45  144", # two full
            "0 0.55 LARM_XYZ_ABS 0.430 -0.230 0.440 -168 -63  131", # two full
            "0 1.00 LARM_XYZ_ABS 0.430  0.260 0.460 -180 -90  145"  # two full
        ]
        self.spin = [
            "0 0.50 LARM_JNT_REL 0 0 0 0 0 0  150", 
            "0 0.50 LARM_JNT_REL 0 0 0 0 0 0 -150"
        ]
    
    def get_empty_action(self):
        open(self.filepath, 'w').close()

    def get_pick_seq(self, xyz, rpy):
        return [
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 1.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
            "0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy)
        ]


    def get_place_seq(self):
        return [
            "0 1.50 LARM_XYZ_ABS %.3f %.3f 0.350 -90.0 -90.0 90.0" % (*self.goal_c,),
            "0 0.50 LARM_XYZ_ABS %.3f %.3f 0.200 -90.0 -90.0 90.0" % (*self.goal_c,),
            "0 0.50 LHAND_JNT_OPEN",
            self.initpose
        ]

    def get_action(self, xyz, rpy, action_idx):
        """Generate helix motion for picking entangled wire harnesses

        Args:
            xyz (tuple or list): left arm position
            rpy (tuple or list): left arm rpy
            action_no (int): 0,1,2,3,4,5,6
        """
        subseqs = [[], self.half_helix, self.half_helix+self.spin,
                self.helix[:7], self.helix[:7]+self.spin,
                self.helix, self.helix+self.spin] 
        
        seqs = self.get_pick_seq(xyz, rpy) + subseqs[action_idx] + self.get_place_seq()
        with open(self.filepath, 'wt') as fp:
            for s in seqs:
                print(s, file=fp)


    
