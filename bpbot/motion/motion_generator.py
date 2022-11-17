import math
from bpbot import BinConfig
class Motion(object):
    def __init__(self, filepath):
        self.filepath = filepath
        # hand open width from the center
        bincfg = BinConfig()
        cfg = bincfg.data
        w_lft = (cfg["hand"]["left"]["open_width"]/2/1000) * 180 / math.pi
        w_rgt = (cfg["hand"]["right"]["open_width"]/2/1000) * 180 / math.pi
        # both hand close
        self.initpose = "0 3.00 JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 0 0 0 0"
        # both hand open 
        self.initpose_ = "JOINT_ABS 0 0 0 -10 -25.7 -127.5 0 0 0 23 -25.7 -127.5 -7 0 0 %.3f %.3f %.3f %.3f" % (w_rgt,-w_rgt,w_lft,-w_lft)
        self.bothhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f %.3f %.3f"% (w_rgt,-w_rgt,w_lft,-w_lft) 
        self.lhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f"% (w_lft,-w_lft) 
        self.rhand_close = "0 0.50 JOINT_REL 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 %.3f %.3f 0 0"% (w_rgt,-w_rgt) 
        self.goal_c = [0.440, 0.310]
        self.drop_c = [0.438, 0.120]
        self.drop_c = [0.438, 0.260]
        self.drop_c = [0.438, 0.200]

        
        self.half_helix = [
            "0 1.00 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90  145", # o
            "0 0.50 LARM_XYZ_ABS 0.600  0.120 0.340 -180 -90  145", # half
            "0 0.50 LARM_XYZ_ABS 0.627 -0.140 0.360 -180 -90  145", # half
            "0 0.75 LARM_XYZ_ABS 0.490 -0.180 0.400 -180 -90  145", # full
            "0 0.75 LARM_XYZ_ABS 0.537  0.260 0.420 -180 -90  145", # o
            # "0 1.00 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90 145",
            # "0 1.00 LARM_XYZ_ABS 0.597  0.060 0.370 -180 -90 145",
            # "0 1.00 LARM_XYZ_ABS 0.497 -0.090 0.370 -180 -90 145",
            # "0 1.00 LARM_XYZ_ABS 0.477  0.260 0.370 -180 -90 145"
        ]
        self.helix = [
            "0 1.00 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90  145", # o
            "0 0.50 LARM_XYZ_ABS 0.600  0.120 0.340 -180 -90  145", # half
            "0 0.50 LARM_XYZ_ABS 0.627 -0.140 0.360 -180 -90  145", # half
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

        self.diagnal = [
            "0 1.00 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90 145", 
            "0 1.50 LARM_XYZ_ABS 0.537 -0.290 0.460  120 -90 145",
            "0 1.50 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90 145" 
        ]
        self.helix_cone = [
            "0 1.00 LARM_XYZ_ABS 0.537  0.160 0.320 -180 -90  145", # o
            "0 0.50 LARM_XYZ_ABS 0.567  0.120 0.340 -180 -90  145", # half
            "0 0.50 LARM_XYZ_ABS 0.587 -0.040 0.360 -180 -90  145", # half
            "0 0.50 LARM_XYZ_ABS 0.527 -0.120 0.380 -180 -90  145", # full
            "0 0.75 LARM_XYZ_ABS 0.490 -0.080 0.400 -180 -90  145", # full
            "0 0.75 LARM_XYZ_ABS 0.437  0.080 0.410 -180 -90  145", # full
            "0 0.75 LARM_XYZ_ABS 0.507  0.160 0.420 -180 -90  145", # o
            "0 0.50 LARM_XYZ_ABS 0.527  0.120 0.430  155 -73 -176", # two full
            "0 0.55 LARM_XYZ_ABS 0.460  0.010 0.440  160 -57  170", # two full
            "0 0.55 LARM_XYZ_ABS 0.430 -0.030 0.440  174 -45  144", # two full
            "0 0.55 LARM_XYZ_ABS 0.400 -0.050 0.440 -168 -63  131", # two full
            "0 1.00 LARM_XYZ_ABS 0.430  0.260 0.460 -180 -90  145"  # two full
        ]

        self.spin = ["0 0.50 LARM_JNT_REL 0 0 0 0 0 0  150", 
                     "0 0.50 LARM_JNT_REL 0 0 0 0 0 0 -150"]
             
    def gen_motion_picking(self, g, dest="goal"):
        # goal_c = [0.086, 0.532]
        [x,y,z,roll,pitch,yaw] = g
        fp=open(self.filepath, 'wt')
        # print("0 0.20 LHAND_JNT_OPEN",file=fp)
        print("0 0.80 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (x,y,roll,pitch,yaw),file=fp)
        print("0 0.50 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (x,y,z,roll,pitch,yaw),file=fp)

        print("0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 0.50 LARM_XYZ_ABS %.3f %.3f 0.300 %.1f %.1f %.1f" % (x,y,roll,pitch,yaw),file=fp)
        if dest == "drop":
            print("0 0.80 LARM_XYZ_ABS %.3f %.3f 0.300 %.1f %.1f %.1f" % (*self.drop_c,roll,pitch,yaw),file=fp)
            print("0 0.50 LARM_XYZ_ABS %.3f %.3f 0.220 %.1f %.1f %.1f" % (*self.drop_c,roll,pitch,yaw),file=fp)
        elif dest == "goal":
            print("0 1.00 LARM_XYZ_ABS 0.086 0.532 0.250 %.1f %.1f %.1f" % (roll,pitch,yaw), file=fp)
            print("0 0.50 LARM_XYZ_ABS 0.086 0.532 0.180 %.1f %.1f %.1f" % (roll,pitch,yaw), file=fp)
            # print("0 0.80 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.goal_c,roll,pitch,yaw), file=fp)
            # print("0 0.30 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*self.goal_c,roll,pitch,yaw), file=fp)
        # elif dest == "side":
        #     print("0 1.00 LARM_XYZ_ABS 0.086 0.532 0.300 %.1f %.1f %.1f" % (roll,pitch,yaw), file=fp)
        #     print("0 1.00 LARM_XYZ_ABS 0.086 0.532 0.200 %.1f %.1f %.1f" % (roll,pitch,yaw), file=fp)
        print("0 0.30 LHAND_JNT_OPEN",file=fp)
        print("0 1.00 "+self.initpose_,file=fp)
        fp.close()

    def gen_motion_test(self, g_pull, v_pull, g_hold):
        [px,py,pz,proll,ppitch,pyaw] = g_pull
        [hx,hy,hz,hroll,hpitch,hyaw] = g_hold
        fp = open(self.filepath, 'wt')
        print("0 3.00 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (px,py,proll,ppitch,pyaw),file=fp)
        print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px,py,pz,proll,ppitch,pyaw),file=fp)
        print("0 3.00 RARM_XYZ_ABS %.3f %.3f 0.200%.1f %.1f %.1f" % (hx,hy,hroll,hpitch,hyaw),file=fp)
        print("0 3.00 RARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (hx,hy,hz,hroll,hpitch,hyaw),file=fp)
        print("0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 0.50 RHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px,py,pz+0.1,proll,ppitch,pyaw),file=fp)
        print("0 3.00 RARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (hx,hy,hz+0.1,hroll,hpitch,hyaw),file=fp)

        # start wiggle
        #for m in self.get_seq_wiggle(px,py,pz+0.1,0.470,0.350,0.250):
        #    print(m, file=fp)

        print("0 2.00 LARM_XYZ_ABS 0.470 0.350 0.250 %.1f %.1f %.1f" % (proll,ppitch,pyaw),file=fp) 
        print("0 3.00 LARM_XYZ_ABS 0.470 0.350 0.200 %.1f %.1f %.1f" % (proll,ppitch,pyaw),file=fp) 
        print(self.bothhand_close, file=fp)
        #print("0 0.50 LHAND_JNT_OPEN",file=fp)
        #print("0 0.50 RHAND_JNT_OPEN",file=fp)
        print("0 3.0 " + self.initpose_,file=fp)

    def gen_motion_test2(self, g_pull, v_pull, g_hold):
        [px,py,pz,proll,ppitch,pyaw] = g_pull
        [hx,hy,hz,hroll,hpitch,hyaw] = g_hold
        [vx,vy,vlen] = v_pull
        fp = open(self.filepath, 'wt')
        print("0 3.00 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (px,py,proll,ppitch,pyaw),file=fp)
        print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px,py,pz,proll,ppitch,pyaw),file=fp)
        print("0 3.00 RARM_XYZ_ABS %.3f %.3f 0.200%.1f %.1f %.1f" % (hx,hy,hroll,hpitch,hyaw),file=fp)
        print("0 3.00 RARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (hx,hy,hz,hroll,hpitch,hyaw),file=fp)
        print("0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 0.50 RHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
        print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px,py,pz+0.1,proll,ppitch,pyaw),file=fp)
        print("0 3.00 RARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (hx,hy,hz+0.1,hroll,hpitch,hyaw),file=fp)

        # start wiggle
        for m in self.get_seq_wiggle(px,py,pz+0.1,0.470,0.350,0.250):
            print(m, file=fp)
        #print("0 2.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px+vlen*vx,py+vlen*vy,pz+0.1,proll,ppitch,pyaw),file=fp)
        #print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px+vlen*vx,py+vlen*vy,pz+0.1,proll,ppitch,pyaw),file=fp)

        print("0 5.00 LARM_XYZ_ABS 0.470 0.350 0.250 %.1f %.1f %.1f" % (proll,ppitch,pyaw),file=fp) 
        print("0 3.00 LARM_XYZ_ABS 0.470 0.350 0.200 %.1f %.1f %.1f" % (proll,ppitch,pyaw),file=fp) 
        print(self.bothhand_close, file=fp)
        #print("0 0.50 LHAND_JNT_OPEN",file=fp)
        #print("0 0.50 RHAND_JNT_OPEN",file=fp)
        print("0 3.0 " + self.initpose_,file=fp)


    def gen_motion_separation(self, g_pull, v_pull, g_hold=None):
        [px,py,pz,proll,ppitch,pyaw] = g_pull
        [vx,vy,vlen] = v_pull
        fp = open(self.filepath, 'wt')

        if g_hold is not None:
            [hx,hy,hz,hroll,hpitch,hyaw] = g_hold
            # print("0 0.50 LHAND_JNT_OPEN",file=fp)
            # print("0 0.50 RHAND_JNT_OPEN",file=fp)
            print("0 1.00 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (px,py,proll,ppitch,pyaw),file=fp)
            print("0 1.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px,py,pz,proll,ppitch,pyaw),file=fp)
            
            print("0 2.00 RARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (hx,hy,hz+0.1,hroll,hpitch,hyaw),file=fp)
            print("0 2.00 RARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (hx,hy,hz,hroll,hpitch,hyaw),file=fp)

            print("0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
            print("0 0.50 RHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)

            # print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px+vlen*vx,py+vlen*vy,pz+0.02,proll,ppitch,pyaw),file=fp)
            print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px+vlen*vx,py+vlen*vy,pz,proll,ppitch,pyaw),file=fp)
            print("0 3.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px+vlen*vx,py+vlen*vy,pz+0.1,proll,ppitch,pyaw),file=fp)

            print("0 2.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.goal_c,proll,ppitch,pyaw),file=fp) 
            print("0 2.00 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*self.goal_c,proll,ppitch,pyaw),file=fp) 
            print(self.bothhand_close, file=fp)
            #print("0 0.50 LHAND_JNT_OPEN",file=fp)
            #print("0 0.50 RHAND_JNT_OPEN",file=fp)
            print("0 0.8 " + self.initpose_,file=fp)

        else: 
            # print("0 0.50 LHAND_JNT_OPEN",file=fp)
            print("0 0.80 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (px,py,proll,ppitch,pyaw),file=fp)
            print("0 0.50 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px,py,pz,proll,ppitch,pyaw),file=fp)
            print("0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",file=fp)
            _e = [px+vlen*vx,py+vlen*vy,pz]
            for m in self.get_seq_wiggle(*g_pull, *_e):
                print(m, file=fp)
            # print("0 0.80 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px+vlen*vx,py+vlen*vy,pz,proll,ppitch,pyaw),file=fp)
            print("0 0.50 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (px+vlen*vx,py+vlen*vy,pz+0.1,proll,ppitch,pyaw),file=fp)


            print("0 1.00 LARM_XYZ_ABS 0.086 0.532 0.300 %.1f %.1f %.1f" % (proll,ppitch,pyaw), file=fp)
            print("0 0.50 LARM_XYZ_ABS 0.086 0.532 0.200 %.1f %.1f %.1f" % (proll,ppitch,pyaw), file=fp)
            # print("0 2.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (*self.goal_c,proll,ppitch,pyaw),file=fp) 
            # print("0 2.00 LARM_XYZ_ABS %.3f %.3f 0.200 %.1f %.1f %.1f" % (*self.goal_c,proll,ppitch,pyaw),file=fp) 
            print("0 0.30 LHAND_JNT_OPEN",file=fp)
            print("0 1.00 " + self.initpose_,file=fp) 

    def get_seq_pick(self,x,y,z,roll,pitch,yaw):
        return [
            # "0 0.50 LHAND_JNT_OPEN",
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (x,y,roll,pitch,yaw),
            "0 1.00 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (x,y,z,roll,pitch,yaw),
            "0 0.50 LHAND_JNT_CLOSE 0 0 0 0 0 0",
            "0 1.00 LARM_XYZ_ABS %.3f %.3f 0.250 %.1f %.1f %.1f" % (x,y,roll,pitch,yaw),
        ]

    def get_seq_place(self,roll,pitch,yaw,destination):
        if destination == "front":
            return [
                # "0 2 LARM_XYZ_ABS 0.48 0.35 0.25 %.1f %.1f %.1f" % (roll,pitch,yaw),
                "0 2.00 LARM_XYZ_ABS 0.480 0.350 0.250 -90.0 -90.0 90.0",
                "0 0.50 LHAND_JNT_OPEN",
                "0 0.80 " + self.initpose_
            ]
        elif destination == "side":
            return [
                # "0 1.00 LARM_XYZ_ABS 0.400 0.400 0.450 -90.0 -90.0 90.0",
                "0 1.50 LARM_XYZ_ABS 0.070 0.552 0.350 -90.0 -90.0 90.0",
                "0 0.50 LARM_XYZ_ABS 0.070 0.552 0.200 -90.0 -90.0 90.0",
                "0 0.50 LHAND_JNT_OPEN",
                "0 0.80 " + self.initpose_
            ]

    def get_seq_wiggle(self,x,y,z,roll,pitch,yaw,fx,fy,fz,freq=8):
        seq = []
        _x = (fx-x)/freq
        _y = (fy-y)/freq
        _z = (fz-z)/freq
        for i in range(freq):
            seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (x+_x*i,y+_y*i,z+_z*i,roll,pitch+3,yaw))
            seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f %.1f %.1f %.1f" % (x+_x*i,y+_y*i,z+_z*i,roll,pitch-3,yaw))
            # seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f -90 -88 -180" % (x+_x*i,y+_y*i,z+_z*i))
            # seq.append("0 0.15 LARM_XYZ_ABS %.3f %.3f %.3f 90 -88 0" % (x+_x*i,y+_y*i,z+_z*i))
        return seq

    def gen_motion_circular(self, pose, sub_action):
        """Generate circular motion for picking entangled wire harnesses        

        Args:
            pose (array): left arm pose [x,y,z,roll,pitch,yaw] 
            sub_action (str): 0,1,2,3,4,5,6
        """
        [x,y,z,roll,pitch,yaw] = pose
        fp = open(self.filepath, 'wt')
        circular_seqs = [[], self.half_helix, self.half_helix+self.spin,
                        self.helix[:7], self.helix[:7]+self.spin,
                        self.helix, self.helix+self.spin]
        seqs = self.get_seq_pick(x,y,z,roll,pitch,yaw) + circular_seqs[sub_action] + ["0 1.00 "+self.initpose_]
        # seqs = self.get_seq_pick(x,y,z,roll,pitch,yaw) + circular_seqs[sub_action] + self.get_seq_place(roll,pitch,yaw,destination="side")
        for m in seqs: 
            print(m, file=fp) 
        fp.close()

    def gen_motion_empty(self):
        open(self.filepath, 'w').close()
    
    def gen_motion_cone_helix(self,x,y,z,roll,pitch,yaw):
        fp = open(self.filepath, 'wt')
        seqs = self.get_seq_pick(x,y,z,roll,pitch,yaw) + self.helix_cone + self.get_seq_place(roll,pitch,yaw,"side")
        for m in seqs:
            print(m, file=fp)
        fp.close()

    def gen_motion_cone_helix_spin(self,x,y,z,roll,pitch,yaw):
        fp = open(self.filepath, 'wt')
        seqs = self.get_seq_pick(x,y,z,roll,pitch,yaw) + self.helix_cone + self.spin + self.get_seq_place(roll,pitch,yaw,"side")
        for m in seqs:
            print(m, file=fp) 
        fp.close()

    def gen_motion_diagnal(self,x,y,z,roll,pitch,yaw):
        fp = open(self.filepath, 'wt')
        seqs = self.get_seq_pick(x,y,z,roll,pitch,yaw) + self.diagnal + self.get_seq_place(roll,pitch,yaw,"side")
        for m in seqs:
            print(m ,file=fp)
        fp.close()
