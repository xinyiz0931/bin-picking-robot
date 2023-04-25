#!/usr/bin/env python
import grpc
import time
import yaml
import math
from concurrent import futures
import nxtrobot_pb2 as nxt_msg
import nxtrobot_pb2_grpc as nxt_rpc
from nextage_ros_bridge import nextage_client
from hrpsys import rtm

# import nxtlib.predefinition.predefinition as pre_def

class NxtServer(nxt_rpc.NxtServicer):
    """
    NOTE: All joint angle parameters are in degrees
    """

    _groups = [['torso', ['CHEST_JOINT0']],
               ['head', ['HEAD_JOINT0', 'HEAD_JOINT1']],
               ['rarm', ['RARM_JOINT0', 'RARM_JOINT1', 'RARM_JOINT2',
                         'RARM_JOINT3', 'RARM_JOINT4', 'RARM_JOINT5']],
               ['larm', ['LARM_JOINT0', 'LARM_JOINT1', 'LARM_JOINT2',
                         'LARM_JOINT3', 'LARM_JOINT4', 'LARM_JOINT5']]]
    # _initpose = [0,0,0,-15,0,-143,0,0,0,15,0,-143,0,0,0]
    _initpose = [0,0,0,-10,-25.7,-127.5,0,0,0,23,-25.7,-127.5,-7,0,0,0.716,-0.716,0.716,-0.716]
    _offpose = OffPose = [0,0,0,25,-140,-150,45,0,0,-25,-140,-150,-45,0,0]

    def _deg2rad(self, degreelist):
        return list(map(math.radians, degreelist))

    def _rad2deg(self, radianlist):
        return list(map(math.degrees, radianlist))
    
    def init(self):
        host = '192.168.128.10'
        port = '15005'
        print('host:' + host)
        print('port:' + port)
        
        rtm.nshost = host
        rtm.nsport = port

        robot = nxc = nextage_client.NextageClient()
        return robot

    def connect(self):
        """
        MUST configure the robot_s in the very beginning
        :return:
        author: weiwei
        date: 20190417
        """
        self._robot = self.init()
        robot_name = "RobotHardware0"
        self._robot.init(robotname=robot_name, url="")
        self._oldyaml = True
        if int(yaml.__version__[0]) >= 5:
            self._oldyaml = False

    def checkEncoders(self, request, context):
        try:
            self._robot.checkEncoders()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def servoOn(self, request, context):
        try:
            self._robot.servoOn()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def servoOff(self, request, context):
        try:
            self._robot.servoOff()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def goInitial(self, request, context):
        try:
            self._robot.goInitial()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def goOffPose(self, request, context):
        try:
            self._robot.goOffPose()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def getJointAngles(self, request, context):
        """
        added by xinyi
        angles in radian
        """
        jntangles = self._robot.getActualState().command # rad
        return nxt_msg.ReturnValue(data = yaml.dump(jntangles))
    
    def getJointPosition(self, request, context):
        """
        added by xinyi
        """
        try:
            if self._oldyaml:
                jnt = yaml.load(request.data)
            else:
                jnt = yaml.load(request.data, Loader=yaml.UnsafeLoader)
            position = self._robot.getCurrentPosition(jnt)
            return nxt_msg.ReturnValue(data = yaml.dump(position))
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def setInitial(self,request,context):
        try:
            if self._oldyaml:
                arm, tm = yaml.load(request.data)
            else:
                arm, tm = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            if tm is None:
                tm = 10.0
            angles = self._initpose
            if arm == 'rarm':
                self._robot.setJointAnglesOfGroup('rarm', angles[3:9], tm, True)
            elif arm == 'larm':
                self._robot.setJointAnglesOfGroup('rarm', angles[9:15], tm, True)
            elif arm == 'all':
                self._robot.goInitial(tm=tm)
            
            return nxt_msg.Status(value = nxt_msg.Status.DONE)

        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)
        
    def setJointAngles(self, request, context):
        """
        :param request: request.data is in degree
        :param context:
        :return:
        author: weiwei
        date: 20190419
        """
        try:
            if self._oldyaml:
                angles, tm, wait = yaml.load(request.data)
            else:
                angles, tm, wait = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            if tm is None:
                tm = 10.0
            self._robot.setJointAnglesOfGroup('torso', angles[0:1], tm, False)
            self._robot.setJointAnglesOfGroup('head', angles[1:3], tm, False)
            self._robot.setJointAnglesOfGroup('rarm', angles[3:9], tm, False)
            # here, set waitInterpolation, revised by xinyi/2023/2/1
            self._robot.setJointAnglesOfGroup('larm', angles[9:15], tm, wait)
            return nxt_msg.Status(value = nxt_msg.Status.DONE)

        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)
    
    def playPattern(self, request, context):
        try:
            if self._oldyaml:
                angleslist, tmlist = yaml.load(request.data)
            else:
                angleslist, tmlist = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            self._robot.playPattern(angleslist, [], [], tmlist)
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def playPatternOfGroup(self, request, context):
        """
        added by xinyi
        @angleslist: degree -> radian in this function
        """
        try:
            if self._oldyaml:
                angleslist, tmlist = yaml.load(request.data)
            else:
                angleslist, tmlist = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            
            gname = 'rarm'
            angleslist_rad = [self._deg2rad(x) for x in angleslist]
            self._robot.playPatternOfGroup(gname, angleslist_rad, tmlist)
            return nxt_msg.Status(value = nxt_msg.Status.DONE)

        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def closeHandToolRgt(self, request, context):
        try:
            self._robot._hands.gripper_r_close()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def closeHandToolLft(self, request, context):
        try:
            self._robot._hands.gripper_l_close()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def openHandToolRgt(self, request, context):
        try:
            self._robot._hands.gripper_r_open()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def openHandToolLft(self, request, context):
        try:
            self._robot._hands.gripper_l_open()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def attachHandToolRgt(self, request, context):
        try:
            self._robot._hands.handtool_r_attach()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def attachHandToolLft(self, request, context):
        try:
            self._robot._hands.handtool_l_attach()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def ejectHandToolRgt(self, request, context):
        try:
            self._robot._hands.handtool_r_eject()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    def ejectHandToolLft(self, request, context):
        try:
            self._robot._hands.handtool_l_eject()
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)

    """This is some additional control commands added by xinyi, 20210801"""

    def setHandAnglesDegRgt(self, request, context):
        """
        add by xinyi
        """
        try:
            if self._oldyaml:
                angles, tm = yaml.load(request.data)
            else:
                angles, tm = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            if tm is None:
                tm = 10.0
            self._robot.setHandJointAngles('rhand', [v * math.pi / 180.0 for v in angles], tm)
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)
            
    def setHandAnglesDegLft(self, request, context):
        """
        add by xinyi
        """
        try:
            if self._oldyaml:
                angles, tm = yaml.load(request.data)
            else:
                angles, tm = yaml.load(request.data, Loader = yaml.UnsafeLoader)
            if tm is None:
                tm = 10.0
            self._robot.setHandJointAngles('lhand', [v * math.pi / 180.0 for v in angles], tm)
            return nxt_msg.Status(value = nxt_msg.Status.DONE)
        except Exception as e:
            print(e, type(e))
            return nxt_msg.Status(value = nxt_msg.Status.ERROR)
    
def serve():
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    nxtserver = NxtServer()
    nxtserver.connect()
    nxt_rpc.add_NxtServicer_to_server(nxtserver, server)
    server.add_insecure_port('[::]:15005')
    server.start()
    print("[*] The Nextage Robot server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    serve()
