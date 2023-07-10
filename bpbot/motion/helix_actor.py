from bpbot.motion import PickAndPlaceActor

class HelixActor(PickAndPlaceActor):
    def __init__(self, filepath):
        """Helix motion is pre-defined only for left arm"""
        self.filepath = filepath
        arm = "L"
        PickAndPlaceActor.__init__(self,filepath, arm)
    
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

    def get_action(self, pose, action_idx):
        """Generate helix motion for picking entangled wire harnesses

        Args:
            xyz (tuple or list): left arm position
            rpy (tuple or list): left arm rpy
            action_no (int): 0,1,2,3,4,5,6
        """
        xyz = pose[:3]
        rpy = pose[3:]
        subseqs = [[], self.half_helix, self.half_helix+self.spin,
                self.helix[:7], self.helix[:7]+self.spin,
                self.helix, self.helix+self.spin] 
        
        seqs = self.get_pick_seq(xyz, rpy) + subseqs[action_idx] + self.get_place_seq(rpy=[-90,-90,90],dest="side")
        with open(self.filepath, 'wt') as fp:
            for s in seqs:
                print(s, file=fp)


    
