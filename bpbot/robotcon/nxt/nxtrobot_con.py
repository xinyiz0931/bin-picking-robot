import grpc
import yaml
import numpy as np
from nxtrobot_client import NxtRobot

class NextageControl(object):
    """
    Customized nextage control policy
    """

    def __init__(self, host = "localhost:18300"):
        self.nxt = NxtRobot(host='[::]:15005')
        self.lift = [
            [2.00, 15.7092, 0, 0, 24.6558, -109.704, -38.7129, 29.6065, 61.5682, -14.5760, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021],
        ]
        self.place = [
            [2.00, -73.1442, 0, 0, 24.0028, -95.363, -57.2524, 33.9753, 64.971, 79.8025, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021],
            [1.00, -73.1442, 0, 0, 18.3841, -64.528, -82.6425, 28.3538, 58.858, 78.9484, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021],
            [0.50, -73.1442, 0, 0, 18.3841, -64.528, -82.6425, 28.3538, 58.858, 78.9484, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0.716021, -0.716021, 0.716021, -0.716021] 

        ]

    def fling(self):
        motions = [
            [1.00, 15.7092, 0, 0, 27.4812, -77.2371, -82.8645,  45.9006, 41.6059, 0.492748, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021],
            [0.53, 15.7092, 0, 0, 27.4812, -77.2371, -82.8645, -39.1019, 41.6059, 0.492748, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021], 
            [0.53, 15.7092, 0, 0, 27.4812, -77.2371, -82.8645,  45.9006, 41.6059, 0.492748, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021], 
            [0.53, 15.7092, 0, 0, 27.4812, -77.2371, -82.8645, -39.1019, 41.6059, 0.492748, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021], 
            [0.53, 15.7092, 0, 0, 27.4812, -77.2371, -82.8645,  45.9006, 41.6059, 0.492748, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021], 
            [0.53, 15.7092, 0, 0, 27.4812, -77.2371, -82.8645, -39.1019, 41.6059, 0.492748, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021], 
            [0.53, 15.7092, 0, 0, 27.4812, -77.2371, -82.8645,  45.9006, 41.6059, 0.492748, 23.0007, -25.7008, -127.504, -7.0002, 0, 0, 0, 0, 0.716021, -0.716021]
        ]
        return motions
        
    def playMotionWithFB(self, motion_seq):
        from bpbot.binpicking import check_force, check_force_file
        import numpy as np
        m_rcvr = np.array([[1,-5.43324762,0,0,-10.00000429,-25.7000219,-127.49972519,0,0,0,-19.17583008,-62.94312106,-86.38696242,-30.22589964,60.94772867,-1.75656712,1.31780293,-1.31780293,0,0],[0.5,-5.43324762,0.,0.,-10.00000429,-25.7000219,-127.49972519,0.,0.,0.,-19.17583008,-62.94312106,-86.38696242,-30.22589964,60.94772867,-1.75656712,1.31780293,-1.31780293,0.97402825,-0.97402825],[0.8,0.,0.,0.,-9.99999792,-25.69999465,-127.49997347,0.,0.,0.,22.99999521,-25.69999465,-127.49997347,-6.99999854,0.,0.,1.31799973,-1.31799973,1.31799973,-1.31799973]])
        """
        added by xinyi: motion_seq shape = (num_seq x 20)
        including both hands open/closing
        """
        try:
            old_lhand = "STANDBY"
            old_rhand = "STANDBY"
            for m in motion_seq[:-4]:
                if (m[-2:] != 0).all(): lhand = "OPEN"
                else: lhand = "CLOSE"
                if (m[-4:-2] != 0).all(): rhand = "OPEN"
                else: rhand = "CLOSE"

                # print(f"left hand: {lhand}, right hand {rhand}")
                if old_rhand != rhand:
                    if rhand == "OPEN": self.nxt.openHandToolRgt()
                    elif rhand == "CLOSE": self.nxt.closeHandToolRgt()

                if old_lhand != lhand:
                    if lhand == "OPEN": self.nxt.openHandToolLft()
                    elif lhand == "CLOSE": self.nxt.closeHandToolLft()
                
                old_lhand = lhand
                old_rhand = rhand

                self.nxt.setJointAngles(m[1:], tm=m[0])

            #is_tangle_b = check_force(1500, 0.1)
            print("*******************")
            is_tangle = check_force_file()
            print("Tangled?", is_tangle)
            print("*******************")

            if is_tangle: 
                for m in m_rcvr:
                    print(m)
                    if (m[-2:] != 0).all(): lhand = "OPEN"
                    else: lhand = "CLOSE"
                    if (m[-4:-2] != 0).all(): rhand = "OPEN"
                    else: rhand = "CLOSE"

                    # print(f"left hand: {lhand}, right hand {rhand}")
                    if old_rhand != rhand:
                        if rhand == "OPEN": self.nxt.openHandToolRgt()
                        elif rhand == "CLOSE": self.nxt.closeHandToolRgt()

                    if old_lhand != lhand:
                        if lhand == "OPEN": self.nxt.openHandToolLft()
                        elif lhand == "CLOSE": self.nxt.closeHandToolLft()
                    
                    old_lhand = lhand
                    old_rhand = rhand

                    self.setJointAngles(m[1:], tm=m[0])
            else: 
                for m in motion_seq[-4:]:
                    if (m[-2:] != 0).all(): lhand = "OPEN"
                    else: lhand = "CLOSE"
                    if (m[-4:-2] != 0).all(): rhand = "OPEN"
                    else: rhand = "CLOSE"

                    # print(f"left hand: {lhand}, right hand {rhand}")
                    if old_rhand != rhand:
                        if rhand == "OPEN": self.nxt.openHandToolRgt()
                        elif rhand == "CLOSE": self.nxt.closeHandToolRgt()

                    if old_lhand != lhand:
                        if lhand == "OPEN": self.nxt.openHandToolLft()
                        elif lhand == "CLOSE": self.nxt.closeHandToolLft()
                    
                    old_lhand = lhand
                    old_rhand = rhand

                    self.nxt.setJointAngles(m[1:], tm=m[0])

        except grpc.RpcError as rpc_error:
            print(f"[!] Robotcon failed with {rpc_error.code()}")
    
    def playMotion(self, motion_seq):
        """
        added by xinyi: motion_seq shape = (num_seq x 20)
        including both hands open/closing
        """
        try:
            old_lhand = "STANDBY"
            old_rhand = "STANDBY"
            for m in motion_seq:
                if (m[-2:] != 0).all(): lhand = "OPEN"
                else: lhand = "CLOSE"
                if (m[-4:-2] != 0).all(): rhand = "OPEN"
                else: rhand = "CLOSE"

                if old_rhand != rhand:
                    if rhand == "OPEN": self.nxt.openHandToolRgt()
                    elif rhand == "CLOSE": self.nxt.closeHandToolRgt()

                if old_lhand != lhand:
                    if lhand == "OPEN": self.nxt.openHandToolLft()
                    elif lhand == "CLOSE": self.nxt.closeHandToolLft()
                
                old_lhand = lhand
                old_rhand = rhand
                
                self.nxt.setJointAngles(m[1:], tm=m[0])

        except grpc.RpcError as rpc_error:
            print(f"[!] Robotcon failed with {rpc_error.code()}")

    def playDynamicMotion(self, motion_seq):

        try:
            old_lhand = "STANDBY"
            old_rhand = "STANDBY"
            for i, m in enumerate(motion_seq):
                if (m[-2:] != 0).all(): lhand = "OPEN"
                else: lhand = "CLOSE"
                if (m[-4:-2] != 0).all(): rhand = "OPEN"
                else: rhand = "CLOSE"

                if old_rhand != rhand:
                    if rhand == "OPEN": self.nxt.openHandToolRgt()
                    elif rhand == "CLOSE": self.nxt.closeHandToolRgt()

                if old_lhand != lhand:
                    if lhand == "OPEN": self.nxt.openHandToolLft()
                    elif lhand == "CLOSE": self.nxt.closeHandToolLft()
                
                old_lhand = lhand
                old_rhand = rhand
                
                self.nxt.setJointAngles(m[1:], tm=m[0])

                with open("/home/hlab/bpbot/data/force/out.txt", 'a') as fp:
                    print(*([-1]*7), file=fp)
        
        except grpc.RpcError as rpc_error:
            print(f"[!] Robocon failed with {rpc_error.code()}")

    def playMotionFT(self, motion_seq):
        # def record():
        #     os.system("bash /home/hlab/bpbot/script/force.sh")
                
            # os.system("bash /home/hlab/bpbot/script/stop_ft.sh")
            # proc_ft.terminate()

        try:
            old_lhand = "STANDBY"
            old_rhand = "STANDBY"
            for i, m in enumerate(motion_seq):
                if (m[-2:] != 0).all(): lhand = "OPEN"
                else: lhand = "CLOSE"
                if (m[-4:-2] != 0).all(): rhand = "OPEN"
                else: rhand = "CLOSE"

                if old_rhand != rhand:
                    if rhand == "OPEN": self.nxt.openHandToolRgt()
                    elif rhand == "CLOSE": self.nxt.closeHandToolRgt()

                if old_lhand != lhand:
                    if lhand == "OPEN": self.nxt.openHandToolLft()
                    elif lhand == "CLOSE": self.nxt.closeHandToolLft()
                
                old_lhand = lhand
                old_rhand = rhand
                
                self.nxt.setJointAngles(m[1:], tm=m[0])

                with open("/home/hlab/bpbot/data/force/out.txt", 'a') as fp:
                    print(*([-1]*7), file=fp)
                # if i == 4:
                #     _data = np.loadtxt("/home/hlab/bpbot/data/force/out.txt")
                #     _idx = np.where(np.any(_data==([-1]*7), axis=1))[0][-2]
                #     ft = _data[_idx+1:-1]
                #     sensor.plot_file(_data=ft)
                # if i == len(motion_seq)-5: 
                #     _data = np.loadtxt("/home/hlab/bpbot/data/force/out.txt")
                #     _idx = np.where(np.any(_data==([-1]*7), axis=1))[0][-2]
                #     ft = _data[_idx+1:-1]
                #     sensor.plot_file(_data=ft)
                # if i == len(motion_seq)-4: 
                #     _data = np.loadtxt("/home/hlab/bpbot/data/force/out.txt")
                #     _idx = np.where(np.any(_data==([-1]*7), axis=1))[0][-2]
                #     ft = _data[_idx+1:-1]
                #     sensor.plot_file(_data=ft)

            # from multiprocessing import Process
            # proc_ft = Process(target=record, args=())
            # proc_nxt = Process(target=move, args=())
            # proc_ft.start()
            # proc_nxt.start()
            # proc_ft.join()
            # proc_nxt.join()
            # proc_ft.start()
            # proc_ft.terminate()
            # print("terminated!")
        except grpc.RpcError as rpc_error:
            print(f"[!] Robocon failed with {rpc_error.code()}")
        


    def playSmoothMotion(self, gname, motion_seq):
        """Given the complete motion_seq and play smoothly, motion must be planned without torso or head joint

        Args:
            gname (str): 4 classes 
                         ['torso': (1) motion_seq[:,1:2]
                          'head' : (2) motion_seq[:,2:4]
                          'rarm' : (6) motion_seq[:,4:10]
                          'larm' : (6) motion_seq[:,10:16]]
            motion_seq (array): Nx20
        """
        tmlist = motion_seq[:,0].tolist()
        if gname == 'torso':
            angleslist = motion_seq[:,1:2]
        elif gname == 'head':
            angleslist = motion_seq[:,2:4]
        elif gname == 'rarm':
            angleslist = motion_seq[:,4:10]
        elif gname == 'larm':
            angleslist = motion_seq[:,10:16]

        returnvalue = self.stub.playPatternOfGroup(nxt_msg.SendValue(data = yaml.dump([angleslist, tmlist]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()


if __name__ == "__main__":
    
    print("hello")
