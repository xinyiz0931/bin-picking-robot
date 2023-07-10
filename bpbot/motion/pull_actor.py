from bpbot.motion import PickAndPlaceActor

class PullActor(PickAndPlaceActor):
    def __init__(self, filepath, arm):
        self.filepath = filepath
        self.arm = arm[0].upper()
        PickAndPlaceActor.__init__(self, filepath, arm)

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
        xyz_u[2] = 0.35

        if wiggle:
            pull_seq = self.get_wiggle_seq(xyz, rpy, xyz_e, 8) + ["0 2.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_u, *rpy)] 
        else:
            pull_seq = [
                "0 3.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_e, *rpy),
                "0 3.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_u, *rpy)
            ]  
        
        seqs = self.get_pick_seq(xyz, rpy)[:-1] + pull_seq + self.get_place_seq(rpy, dest="side")
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
            seq.append("0 0.15 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*_xyz, *rpy_bfr))
            seq.append("0 0.15 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*_xyz, *rpy_aft))
        return seq
    