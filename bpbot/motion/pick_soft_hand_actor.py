from bpbot.motion import PickAndPlaceActor

class PickSoftHandActor(PickAndPlaceActor):
    def __init__(self, filepath, cfgdata, arm='L'):
        """Helix motion is pre-defined only for left arm"""
        self.filepath = filepath
        self.cfgdata = cfgdata
        PickAndPlaceActor.__init__(self,filepath, arm)
    
    def get_dip_seq(self, water='hot'):
        if water == 'hot':
            xyz = self.cfgdata['position']['hot_water']
        elif water == 'cold':
            xyz = self.cfgdata['position']['cold_water']
        return [
            "0 0.50 "+self.arm+"HAND_JNT_OPEN",
            "0 2.00 LARM_XYZ_ABS %.3f %.3f 0.250 -90.0 -90.0 90.0" % (*xyz,),
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.180 -90.0 -90.0 90.0" % (*xyz,),
            "0 2.00 LARM_XYZ_ABS %.3f %.3f 0.180 -90.0 -90.0 90.0" % (*xyz,),
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.250 -90.0 -90.0 90.0" % (*xyz,),
        ]
    
    def get_grasp_seq(self, xyz, rpy): 
        return [
            "0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz[:2], xyz[2]+0.02, *rpy),
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
            "1 PAUSE",
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
        ]

    def get_place_seq(self, xyz, rpy):
        return [
            "0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz,*rpy),
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*xyz,*rpy),
            "0 0.50 "+self.arm+"HAND_JNT_OPEN",
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz,*rpy),
            self.initpose(3)
        ]

    def get_action(self, pose, rigid=True):
        xyz = pose[:3]
        rpy = pose[3:]
        
        place_xyz = self.cfgdata['position']['place']

        if rigid:
            seqs = self.get_dip_seq('cold')+ self.get_grasp_seq(xyz, rpy) + self.get_place_seq(place_xyz, rpy)
        else: 
            seqs = self.get_dip_seq('hot')+ self.get_grasp_seq(xyz, rpy) + self.get_place_seq(place_xyz, rpy)
        with open(self.filepath, 'wt') as fp:
            for s in seqs:
                print(s, file=fp)


    
