import numpy as np
import math
from scipy import interpolate
from bpbot import BinConfig

class FlingActor(object):
    def __init__(self, arm="R", filepath=None):
        
        cfg = BinConfig()
        cfgdata = cfg.data
        
        if filepath is None: 
            self.filepath = cfg.motionfile_path
        else:
            self.filepath = filepath 
        self.arm = arm.upper()
        
        w_lft = (cfgdata["hand"]["left"]["open_width"]/2/1000) * 180 / math.pi
        w_rgt = (cfgdata["hand"]["right"]["open_width"]/2/1000) * 180 / math.pi
        
        self.initpose = "0 0.80 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 %.3f %.3f %.3f %.3f" % (w_rgt,-w_rgt,w_lft,-w_lft)
        self.bothhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f %.3f %.3f"% (w_rgt,-w_rgt,w_lft,-w_lft) 
        self.lhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f"% (w_lft,-w_lft) 
        self.rhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f 0 0"% (w_rgt,-w_rgt) 
       
        self.goal_c = [0.480, 0.350]
        self.drop_c = [0.438, 0.200]

        self.waypoint = [
            [0.500, 0.400],
            [0.510, 0.420],
            [0.540, 0.300]
        ]

    def fit_ellipse(self, p1, p2, freq=24):
        u, v = p1[0], p2[1]
        print(u,v  )
        a,b = np.abs(np.array(p1)-np.array(p2))
        # t = np.linspace(0, math.pi/2, freq)
        t = np.linspace(3*math.pi/2, 2*math.pi, freq)
        xnew = u+a*np.cos(t)
        ynew = v+b*np.sin(t)
        import matplotlib.pyplot as plt
        plt.plot(xnew, ynew, color='b')
        plt.plot([p1[0],p2[0]],[p1[1], p2[1]], 'o', color='r')
        plt.legend(['Trajectory', 'Waypoints'])
        plt.xlim([0.4,0.6])
        plt.show()
        return xnew, ynew

    def fit_spline(self, p1, p2, p3, freq=24):
        x, y = [], []
        for p in [p1,p2,p3]:
            x.append(p[0])
            y.append(p[1])
        # itvl = (max(x) - min(x)) / freq
        xnew = np.linspace(min(x), max(x), freq)
        # xnew = np.arange(min(x), max(x), itvl)

        tck = interpolate.splrep(x, y, s=0, k=2)
        ynew = interpolate.splev(xnew, tck, der=0)
        print(xnew.shape, ynew.shape)
        import matplotlib.pyplot as plt
        plt.plot(xnew, ynew, color='b')
        plt.plot(x,y, 'o', color='r')
        # plt.plot(xnew, ynew, x,y, 'o')
        plt.legend(['Trajectory', 'Waypoints'])
        plt.xlim([0.4, 0.6])
        plt.title('Cubic-spline interpolation')
        plt.show()
        return xnew, ynew

    def get_pick_seq(self, xyz, rpy): 
        return [
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
            "0 0.50 "+self.arm+"HAND_JNT_CLOSE 0 0 0 0 0 0", 
            "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy)
        ]
    
    def get_place_seq(self, rpy, dest="side"):
        if dest == "front":
            return [
                "0 0.80 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.300 %.1f %.1f %.1f" % (*self.drop_c, *rpy),
                "0 0.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.drop_c, *rpy),
                "0 0.50 "+self.arm+"HAND_JNT_OPEN",
                self.initpose
            ]
        elif dest == "side":
            return [
                "0 1.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.350 %.1f %.1f %.1f" % (*self.goal_c, *rpy),
                "0 0.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*self.goal_c, *rpy),
                "0 0.50 "+self.arm+"HAND_JNT_OPEN",
                self.initpose
            ]

    def get_fling_seq(self, waypoint=None, itvl=16):
        """Get pulling motionfile command

        Args:
            xyz (tuple or list): [x,y,z]
            rpy (tuple or list): [r,p,y]
            wiggle (bool, optional): pulling with wiggling?. defaults to False.
        """
        p1, p2, p3 = self.waypoint if waypoint is None else waypoint
        

        p1 = [0.500, 0.400]
        p2 = [0.510, 0.420]
        p3 = [0.540, 0.300]
        n = 16
        tm = 3/n
        fitted_x, fitted_z = self.fit_spline(p1, p2, p3, freq=n)
        seqs = []
        for x, z in zip(fitted_x, fitted_z):
            seqs.append("0 %.2f " % (tm,)+self.arm+"ARM_XYZ_ABS %.3f -0.010 %.3f 180.0 -80 -180" % (x, z))
        return seqs

    def get_action(self, pose, waypoint=None):
        xyz = pose[:3]
        rpy = pose[3:]

        seqs = self.get_pick_seq(xyz, rpy) + self.get_fling_seq(waypoint) + self.get_place_seq(rpy)
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

<<<<<<< HEAD
actor=FlingActor()
        # x = np.array([0.5,0.51,0.54])
        # y = np.array([0.4,0.42, 0.3])
# actor.get_action([0.500, -0.010, 0.104, -90, -90, 90])
p = [
    [0.500, 0.430],
    [0.510, 0.450],
    [0.540, 0.250]
]
actor.get_fling_seq(waypoint=p)
=======
actor=FlingActor("./data/motion/fling.dat")
p1 = [0.500, 0.400]
p2 = [0.510, 0.420]
p3 = [0.540, 0.300]
actor.fit_spline(p1, p2, p3)
p1 = [0.500, 0.300]
p2 = [0.540, 0.420]
actor.fit_ellipse(p1=p1, p2=p2)

# actor.get_action([0.500, -0.010, 0.104, -90, -90, 90])
>>>>>>> 6a420999ea9af2578d32bbee2ca8138469e84dc3
