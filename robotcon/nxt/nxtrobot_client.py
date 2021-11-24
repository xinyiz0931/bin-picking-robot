import grpc
import sys
sys.path.append("./")
import argparse
from utils.base_utils import *
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

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('movement',type=str, 
                        help='movement you want')
    args = parser.parse_args()
    nxt = NxtRobot(host='[::]:15005')

    if args.movement == "recovery":
        result_print("Recovery from emergency ... ")
        nxt.servoOff()
        nxt.servoOn()
        nxt.goInitial()

    elif args.movement == "startrobot":
        result_print("Start the nextage robot ... ")
        nxt.checkEncoders()
        nxt.goInitial()
    
    elif args.movement == "restartrobot":
        result_print("Restart the nextage robot ... ")
        nxt.servoOn()
        nxt.goInitial()

    elif args.movement == "goinitial":
        result_print("Start going intial pose ... ")
        nxt.goInitial()
    
    elif args.movement == "stoprobot":
        warning_print("Stop the nextage robot ... ")
        nxt.goOffPose()
    elif args.movement == "closehand":
        warning_print("Close hand ... ")
        nxt.closeHandToolLft()
    elif args.movement == "openhand":
        warning_print("Open hand ... ")
        nxt.openHandToolLft() 
    else:
        warning_print("Wrong robot movment! ")

