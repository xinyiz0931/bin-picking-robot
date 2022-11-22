import grpc
import sys
import argparse
import yaml

import nxtrobot_pb2 as nxt_msg
import nxtrobot_pb2_grpc as nxt_rpc

class NxtRobot(object):

    def __init__(self, host = "localhost:18300"):
        channel = grpc.insecure_channel(host)
        self.stub = nxt_rpc.NxtStub(channel)
        self._oldyaml = True
        if int(yaml.__version__[0]) >= 5:
            self._oldyaml = False

    def checkEncoders(self):
        returnvalue = self.stub.checkEncoders(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("Encoders succesfully checked.")

    def servoOn(self):
        returnvalue = self.stub.servoOn(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("Servos are turned on.")

    def servoOff(self):
        returnvalue = self.stub.servoOff(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("Servos are turned off.")

    def goInitial(self):
        returnvalue = self.stub.goInitial(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to its initial pose.")

    def goOffPose(self):
        returnvalue = self.stub.goOffPose(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the off pose.")
    
    def getJointPosition(self, jnt):
        if self._oldyaml:
            position = yaml.load(self.stub.getJointPosition(nxt_msg.SendValue(data=yaml.dump(jnt))).data)
        else:
            position = yaml.load(self.stub.getJointPosition(nxt_msg.SendValue(data=yaml.dump(jnt))).data, Loader=yaml.UnsafeLoader)
        return position


    def getJointAngles(self):
        if self._oldyaml:
            jntangles = yaml.load(self.stub.getJointAngles(nxt_msg.Empty()).data)
        else:
            jntangles = yaml.load(self.stub.getJointAngles(nxt_msg.Empty()).data, Loader=yaml.UnsafeLoader)
        return jntangles

    def setJointAngles(self, angles, tm = None):
        """
        All angles are in degree
        The tm is in second
        :param angles: [degree]
        :param tm: None by default
        :return:
        author: weiwei
        date: 20190417
        """
        returnvalue = self.stub.setJointAngles(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        # else:
        #     print("The robot_s is moved to the given pose.")

    def playPattern(self, angleslist, tmlist = None):
        """
        :param angleslist: [[degree]]
        :param tm: [second]
        :return:
        author: weiwei
        date: 20190417
        """
        if tmlist is None:
            tmlist = [.3]*len(angleslist)
        returnvalue = self.stub.playPattern(nxt_msg.SendValue(data = yaml.dump([angleslist, tmlist]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has finished the given motion.")

    def closeHandToolRgt(self):
        returnvalue = self.stub.closeHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has closed its right handtool.")

    def closeHandToolLft(self):
        returnvalue = self.stub.closeHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has closed its left handtool.")

    def openHandToolRgt(self):
        returnvalue = self.stub.openHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has opened its right handtool.")

    def openHandToolLft(self):
        returnvalue = self.stub.openHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has opened its left handtool.")

    def attachHandToolRgt(self):
        returnvalue = self.stub.attachHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has attached its right handtool.")

    def attachHandToolLft(self):
        returnvalue = self.stub.attachHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has attached its left handtool.")

    def ejectHandToolRgt(self):
        returnvalue = self.stub.ejectHandToolRgt(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has ejected its right handtool.")

    def ejectHandToolLft(self):
        returnvalue = self.stub.ejectHandToolLft(nxt_msg.Empty()).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s has ejected its left handtool.")
    
    
    """This is some additional control commands added by xinyi, 20210801"""

    def setHandAnglesDegRgt(self, angles, tm = None):
        """
        add by xinyi
        """
        returnvalue = self.stub.setHandAnglesDegRgt(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the given pose.")
    def setHandAnglesDegLft(self, angles, tm = None):
        """
        add by xinyi
        """
        returnvalue = self.stub.setHandAnglesDegLft(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the given pose.")

    def moveArmRelRgt(self, angles, tm = None):
        """
        add by xinyi
        """
        returnvalue = self.stub.moveArmRelRgt(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the given pose.")

    def moveArmRelLft(self, angles, tm = None):
        """
        add by xinyi
        """
        returnvalue = self.stub.moveArmRelLft(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the given pose.")

    def moveArmAbsRgt(self, angles, tm = None):
        """
        add by xinyi
        """
        returnvalue = self.stub.moveArmAbsRgt(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the given pose.")

    def moveArmAbsLft(self, angles, tm = None):
        """
        add by xinyi
        """
        returnvalue = self.stub.moveArmAbsLft(nxt_msg.SendValue(data = yaml.dump([angles, tm]))).value
        if returnvalue == nxt_msg.Status.ERROR:
            print("Something went wrong with the server!! Try again!")
            raise Exception()
        else:
            print("The robot_s is moved to the given pose.")
    
    def playMotionSeqWithFB(self, motion_seq):
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
                    if rhand == "OPEN": self.openHandToolRgt()
                    elif rhand == "CLOSE": self.closeHandToolRgt()

                if old_lhand != lhand:
                    if lhand == "OPEN": self.openHandToolLft()
                    elif lhand == "CLOSE": self.closeHandToolLft()
                
                old_lhand = lhand
                old_rhand = rhand

                self.setJointAngles(m[1:], tm=m[0])

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
                        if rhand == "OPEN": self.openHandToolRgt()
                        elif rhand == "CLOSE": self.closeHandToolRgt()

                    if old_lhand != lhand:
                        if lhand == "OPEN": self.openHandToolLft()
                        elif lhand == "CLOSE": self.closeHandToolLft()
                    
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
                        if rhand == "OPEN": self.openHandToolRgt()
                        elif rhand == "CLOSE": self.closeHandToolRgt()

                    if old_lhand != lhand:
                        if lhand == "OPEN": self.openHandToolLft()
                        elif lhand == "CLOSE": self.closeHandToolLft()
                    
                    old_lhand = lhand
                    old_rhand = rhand

                    self.setJointAngles(m[1:], tm=m[0])

        except grpc.RpcError as rpc_error:
            print(f"[!] Robotcon failed with {rpc_error.code()}")

    def playMotionSeq(self, motion_seq):
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

                # print(f"left hand: {lhand}, right hand {rhand}")
                if old_rhand != rhand:
                    if rhand == "OPEN": self.openHandToolRgt()
                    elif rhand == "CLOSE": self.closeHandToolRgt()

                if old_lhand != lhand:
                    if lhand == "OPEN": self.openHandToolLft()
                    elif lhand == "CLOSE": self.closeHandToolLft()
                
                old_lhand = lhand
                old_rhand = rhand

                self.setJointAngles(m[1:], tm=m[0])
        except grpc.RpcError as rpc_error:
            print(f"[!] Robotcon failed with {rpc_error.code()}")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('movement',type=str, 
                        help='movement you want')
    args = parser.parse_args()
    nxt = NxtRobot(host='[::]:15005')

    if args.movement == "recovery":
        print("Recovery from emergency ... ")
        nxt.servoOff()
        nxt.servoOn()
        nxt.goInitial()

    elif args.movement == "startrobot":
        print("Start the nextage robot ... ")
        nxt.checkEncoders()
        nxt.goInitial()
    
    elif args.movement == "restartrobot":
        print("Restart the nextage robot ... ")
        nxt.servoOn()
        nxt.goInitial()

    elif args.movement == "goinitial":
        print("Start going intial pose ... ")
        nxt.goInitial()
    
    elif args.movement == "stoprobot":
        print("Stop the nextage robot ... ")
        nxt.goOffPose()
    elif args.movement == "restartrobot":
        print("Servo on and restart")
        nxt.servoOn()
        nxt.goinitial()

    elif args.movement == "closelhand":
        print("Close left hand ... ")
        nxt.closeHandToolLft()
    elif args.movement == "openlhand":
        print("Open left hand ... ")
        nxt.openHandToolLft() 
    elif args.movement == "closerhand":
        print("Close right hand ... ")
        nxt.closeHandToolRgt()
    elif args.movement == "openrhand":
        print("Open right hand ... ")
        nxt.openHandToolRgt() 
    
    elif args.movement == "tmp":
        jnt  = "LARM_JOINT5"
        print("Test code: get position for", jnt)
        print(nxt.getJointPosition(jnt))
    else:
        print("Wrong robot movment! ")

