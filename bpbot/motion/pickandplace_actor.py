import math
from bpbot import BinConfig

class PickAndPlaceActor(object):
    def __init__(self, filepath, arm='R'):
        
        # cfg = BinConfig()
        # cfgdata = cfg.data
        # if filepath is None: 
        #     self.filepath = cfg.motionfile_path
        # else:
        #     self.filepath = filepath 
        
        # w_lft = (cfgdata["hand"]["left"]["open_width"]/2/1000) * 180 / math.pi
        # w_rgt = (cfgdata["hand"]["right"]["open_width"]/2/1000) * 180 / math.pi
        # self.bothhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f %.3f %.3f"% (w_rgt,-w_rgt,w_lft,-w_lft) 
        # self.lhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f"% (w_lft,-w_lft) 
        # self.rhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f 0 0"% (w_rgt,-w_rgt) 
        
        # self.initpose = "0 1.00 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 %.3f %.3f %.3f %.3f" % (w_rgt,-w_rgt,w_lft,-w_lft)
        # self.initpose = "0 1.00 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 1.375, -1.375, 1.375, -1.375")
        lambda a : a + 10
        self.initpose = lambda tm: "0 %.2f JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 0 0 0 0" % tm
        self.filepath = filepath
        self.arm = arm[0].upper()

        if self.arm == "L":
            self.dest_front_loc = [0.430, 0.200]
            self.dest_side_loc = [0.000, 0.552]
        elif self.arm == "R":
            self.dest_loc = [0.430, -0.200]
            self.dest_side_loc = [0.000, -0.552]

    def gen_empty_action(self):
        open(self.filepath, 'w').close()
    
    def get_grasp_seq(self, xyz, rpy): 
        return [
            "0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz[:2], xyz[2]+0.02, *rpy),
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
        ]

    def get_pick_seq(self, xyz, rpy):
        buffer_x = 0.43 if xyz[0] < 0.4 else xyz[0]
        return [
            "0 0.50 "+self.arm+"HAND_JNT_OPEN",
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 0.80 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz[:2], xyz[2]+0.02, *rpy),
            "0 0.80 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
            "0 0.50 "+self.arm+"HAND_JNT_CLOSE 0 0 0 0 0 0",
            "0 0.80 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz[:2], xyz[2]+0.02, *rpy),
            "0 0.80 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.350 %.1f %.1f %.1f" % (buffer_x, xyz[1], *rpy),
        ]

    def get_place_seq(self, rpy, dest):
        if dest == "front":
            return [
                "0 0.80 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.300 %.1f %.1f %.1f" % (*self.dest_front_loc,*rpy),
                "0 0.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.dest_front_loc,*rpy),
                "0 0.50 "+self.arm+"HAND_JNT_OPEN",
                self.initpose(1)
            ]

        elif dest == "side":
            return [
                # "0 1.50 RARM_XYZ_ABS %.3f %.3f 0.350 -90.0 -90.0 90.0" % (*self.goal_c,),
                # "0 0.50 RARM_XYZ_ABS %.3f %.3f 0.200 -90.0 -90.0 90.0" % (*self.goal_c,),
                "0 1.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.350 %.1f %.1f %.1f" % (*self.dest_side_loc,*rpy),
                "0 0.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.dest_side_loc,*rpy),
                "0 0.50 "+self.arm+"HAND_JNT_OPEN",
                self.initpose(1)
            ]
     
    def get_action(self, pose, dest="front"):
        """Get action for pick-and-place motion

        Args:
            pose (tuple or list): grasp pose
            dest (str, optional): destination, "front", "side". Defaults to "front".
        """
        xyz = pose[:3]
        rpy = pose[3:]
        

        seqs = self.get_pick_seq(xyz, rpy) + self.get_place_seq(rpy, dest=dest)
        with open(self.filepath, 'wt') as fp:
            for s in seqs:
                print(s, file=fp)
