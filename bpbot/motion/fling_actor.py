import numpy as np
import math
from scipy import interpolate
from bpbot import BinConfig

class FlingActor(object):
    def __init__(self, filepath=None, arm='R'):
        
        cfg = BinConfig()
        cfgdata = cfg.data
        
        if filepath is None: 
            self.filepath = cfg.motionfile_path
        else:
            self.filepath = filepath 
        self.arm = arm.upper()
        
        w_lft = (cfgdata["hand"]["left"]["open_width"]/2/1000) * 180 / math.pi
        w_rgt = (cfgdata["hand"]["right"]["open_width"]/2/1000) * 180 / math.pi
        
        self.initpose = "0 3.00 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 %.3f %.3f %.3f %.3f" % (w_rgt,-w_rgt,w_lft,-w_lft)
        self.bothhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f %.3f %.3f"% (w_rgt,-w_rgt,w_lft,-w_lft) 
        self.lhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f"% (w_lft,-w_lft) 
        self.rhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f 0 0"% (w_rgt,-w_rgt) 

        self.wrist_y_limit = 263 # unit: deg/s, horizontal fling
        self.wrist_p_limit = 224 # unit: deg/s, vertically fling
        self.wrist_y_index = 3
        self.wrist_p_index = 4
       
        self.front_bin = [0.438, 0.200]
        self.right_bin = [0.000, -0.500]
        self.left_bin = [0.000, 0.550]

        self.waypoint = [
            [0.450, 0.500],
            [0.460, 0.520],
            [0.490, 0.300]
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
            "0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*xyz[:2], *rpy),
            "0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz, *rpy),
            "0 1.00 "+self.arm+"HAND_JNT_CLOSE 0 0 0 0 0 0",
            "0 3.00 RARM_XYZ_ABS 0.480 -0.010 0.550 -90.0 -90.0 90.0",
            "0 PAUSE",]
        # ] + self.get_wiggle_seq(xyz_s=[0.480, -0.010, xyz[2]], xyz_e=[0.480, -0.010, 0.550], rpy=[-90.0, -90.0, 90.0])
    
    def get_place_seq(self, rpy=None, dest="right"):
        if rpy is None:
            rpy = [-90.0, -90.0, 90.0]
        if dest == "front":
            return [
                "0 0.80 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.300 %.1f %.1f %.1f" % (*self.front_bin, *rpy),
                "0 0.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.front_bin, *rpy),
                "0 0.50 "+self.arm+"HAND_JNT_OPEN",
                self.initpose
            ]
        elif dest == "right":
            return [
                "0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.500 %.1f %.1f %.1f" % (*self.right_bin, *rpy),
                "0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.350 %.1f %.1f %.1f" % (*self.right_bin, *rpy),
                self.rhand_close if self.arm == 'R' else self.lhand_close,
                # "0 0.50 "+self.arm+"HAND_JNT_OPEN",
                self.initpose
            ]
        elif dest == "left":
            return [
                "0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.350 %.1f %.1f %.1f" % (*self.left_bin, *rpy),
                "0 0.50 "+self.arm+"ARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*self.left_bin, *rpy),
                "0 0.50 "+self.arm+"HAND_JNT_OPEN",
                self.initpose
            ]

    # def get_fling_seq(self, waypoint=None, freq=16, n=1):
    #     """Get pulling motionfile command

    #     Args:
    #         xyz (tuple or list): [x,y,z]
    #         rpy (tuple or list): [r,p,y]
    #         wiggle (bool, optional): pulling with wiggling?. defaults to False.
    #     """
    #     p1, p2, p3 = self.waypoint if waypoint is None else waypoint

    #     # tm = 3/freq
    #     tm = 0.4
    #     fitted_x, fitted_z = self.fit_spline(p1, p2, p3, freq=freq)
    #     seqs = []
    #     for i, (x, z) in enumerate(zip(fitted_x, fitted_z)):
    #         if i == 0:
    #             seqs.append("0 2.00 "+self.arm+"ARM_XYZ_ABS %.3f -0.010 %.3f 180.0 -80.0 -180.0" % (x, z))
    #         elif i <=9: 
    #             tm = 0.12
    #             seqs.append("0 %.2f " % (tm,)+self.arm+"ARM_XYZ_ABS %.3f -0.010 %.3f 180.0 -80.0 -180.0" % (x, z))
    #         else:
    #             tm = 0.1
    #             seqs.append("0 %.2f " % (tm,)+self.arm+"ARM_XYZ_ABS %.3f -0.010 %.3f 180.0 -80.0 -180.0" % (x, z))
    #     # return seqs
    #     return seqs*n

    def get_action(self, pose, waypoint=None, orientation='h'):
        xyz = pose[:3]
        rpy = pose[3:]
        # seqs = self.get_pick_seq(xyz, rpy) + self.get_fling_seq(waypoint) + self.get_place_seq(rpy)
        # seqs = self.get_pick_seq(xyz, rpy) + self.get_fling_seq(waypoint, n=3)
        # seqs = self.get_pick_seq(xyz, rpy)
        seqs = self.get_pick_seq(xyz, rpy) + self.get_fling_seq() + self.get_place_seq(dest="right")
        
        with open(self.filepath, 'wt') as fp:
            for s in seqs:
                print(s, file=fp)
    
    def add_fling_action(self, j3, j4, h):
        seqs = self.get_fling_seq(j3=j3, j4=j4, h=h) + self.get_place_seq(dest="right")
            # seqs = self.get_place_seq(dest="right")
        with open(self.filepath, 'a') as fp:
            for s in seqs:
                print(s, file=fp)
    
    def add_place_action(self, dest="right"):
        seqs = self.get_place_seq(dest=dest)
        with open(self.filepath, 'a') as fp:
            for s in seqs:
                print(s, file=fp)

    
    def get_fling_seq(self, j3=60, j4=60, v=120, freq=2, h=0.48):
        tm = max(np.abs(j3)/v, np.abs(j4)/v)
        seq = [
            "0 1.00 "+self.arm+"ARM_XYZ_ABS 0.480 -0.010 %.3f -180.0 -60.0 180.0" % (h,),
            "0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, -j3, -j4)
        ]
        for i in range(freq):
            if i == freq-1:
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, j3, j4))
            else:
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, j3, j4))
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, -j3, -j4))
        seq.append("0 3.00 RARM_XYZ_ABS 0.350 -0.350 0.500 -90.0 -90.0 90.0")
        return seq

    def get_shake_h_seq(self, angle=35, v=160, freq=3):
        tm = angle/v
        seq = [
            "0 1.00 "+self.arm+"ARM_XYZ_ABS 0.480 -0.010 0.500 -180.0 -60.0 180.0",
            "0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0 0" % (tm, -angle)
        ]
        # for i in range(freq):
        #     if i == freq-1:
        #         seq.append("0 0.40 JOINT_REL 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0 0" % (-2*angle))
        #     else:
        #         seq.append("0 0.40 JOINT_REL 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0 0" % (-2*angle))
        #         seq.append("0 0.40 JOINT_REL 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0 0" % (2*angle))

        for i in range(freq):
            if i == freq-1:
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0 0" % (tm, angle))
            else:
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0 0" % (tm, angle))
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0 0" % (tm, -angle))
        return seq
        
    def get_shake_v_seq(self, angle=50, v=160, freq=3):
        tm = angle/v
        seq = [
            "0 1.00 "+self.arm+"ARM_XYZ_ABS 0.480 -0.010 0.330 -90.0 -90.0 90.0",
            "0 %.2f JOINT_REL 0 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, -angle)
        ]
        for i in range(freq):
            if i == freq-1:
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, angle))
            else:
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, angle))
                seq.append("0 %.2f JOINT_REL 0 0 0 0 0 0 0 %.3f 0 0 0 0 0 0 0 0 0 0 0" % (tm, -angle))
        return seq

    def get_wiggle_seq(self, xyz_s, xyz_e, rpy, freq=3):
        seq = ["0 1.00 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_s, *rpy)]

        rpy_bfr, rpy_aft = rpy.copy(), rpy.copy()
        # rpy_bfr[1] += 3
        # rpy_aft[1] -= 3
        rpy_bfr[1] += 3
        rpy_aft[1] -= 3
        for i in range(freq):
            delta_xyz = [xyz_s[k]+(xyz_e[k]-xyz_s[k])*(i*2+1)/(freq*2+1) for k in range(3)]
            delta2_xyz = [xyz_s[k]+(xyz_e[k]-xyz_s[k])*(i*2+2)/(freq*2+1) for k in range(3)]
            # seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*_xyz, *rpy_bfr))
            # seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*_xyz, *rpy_aft))
            seq.append("0 0.66 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*delta_xyz, *rpy_bfr))
            seq.append("0 0.66 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*delta2_xyz, *rpy_aft))
        seq.append("0 0.66 "+self.arm+"ARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (*xyz_e, *rpy))
        seq.append("0 PAUSE")
        return seq

if __name__ == '__main__':

    actor=FlingActor(arm="R")
    # A = actor.get_fling_seq(j4=60, j5=60, freq=2)
    

    actor.get_action([0.500, -0.010, 0.184, -90, -90, 90])

    # p1 = [0.500, 0.400]
    # p2 = [0.510, 0.420]
    # p3 = [0.540, 0.300]
    # actor.fit_spline(p1, p2, p3)
    # p1 = [0.500, 0.300]
    # p2 = [0.540, 0.420]
    # actor.fit_ellipse(p1=p1, p2=p2)
