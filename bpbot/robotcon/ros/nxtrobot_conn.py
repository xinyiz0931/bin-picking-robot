import os
import time
from math import pi

from numpy.lib.function_base import angle

from hironx_ros_bridge.ros_client import ROS_Client
from nextage_ros_bridge import nextage_client
from hrpsys import rtm

def init():
    global robot
    host = '192.168.128.10'
    port = '15005'
    print('host:' + host)
    print('port:' + port)
    robot_name = "RobotHardware0"
    rtm.nshost = host
    rtm.nsport = port
    robot = nxc = nextage_client.NextageClient()
    return robot

def connect():
    robot_name = "RobotHardware0" #"HiroNX(Robot)0"
    robot.init(robotname=robot_name, url="")
    print('Robot connected! ')

def servoON():
    robot.servoOn()
    
def servoOFF():
    robot.servoOff()
    
def calibrateJoint():
    robot.checkEncoders()
    
def goInitial():
    robot.goInitial()
    
def goOffPose():
    robot.goOffPose()
    
def rhandOpen(distance=100):
    print('rhandOpen')
    robot._hands.airhand_r_release()
    
def rhandClose(distance=0):
    print('rhandClose')
    i=0
    while(i<100):
        robot._hands.airhand_r_drawin()
        i=i+1
        time.sleep(0.2)
        
def lhandOpen(distance=100):
    print('lhandOpen1')
    robot._hands.gripper_l_open()
    
def lhandClose(distance=0):
    print('lhandClose1')
    robot._hands.gripper_l_close()

def rhandAttach():
    print('attach')
    robot._hands.handtool_r_attach()
    
def rhandDetach():
    print('detach')
    robot._hands.handtool_r_eject()
    
def lhandAttach():
    print('Attach')
    robot._hands.handtool_l_attach()
    
def lhandDetach():
    print('Detach')
    robot._hands.handtool_l_eject() 

    
def setRHandAnglesDeg(angles):
    robot.setHandJointAngles('rhand', [v * pi / 180.0 for v in angles], 1.0)
    
def setLHandAnglesDeg(angles):
    robot.setHandJointAngles('lhand', [v * pi / 180.0 for v in angles], 1.0)

def moveRArmRel(dx, dy, dz, dr, dp, dw):
    robot.setTargetPoseRelative('rarm', 'RHAND_JOINT5', dx, dy, dz, dr, dp, dw, 1.0)
    
def moveLArmRel(dx, dy, dz, dr, dp, dw):
    robot.setTargetPoseRelative('larm', 'LHAND_JOINT5', dx, dy, dz, dr, dp, dw, 1.0)

def moveLArmAbs(x,y,z,r,p,w):
    angles = [r,p,w]
    robot.setTargetPose('larm',[x,y,z], [v * pi / 180.0 for v in angles], 2.0)

def setJointAngles(angles,tm=1.0):
    robot.setJointAnglesOfGroup('torso', angles[0:1], tm, False)
    robot.setJointAnglesOfGroup('head', angles[1:3], tm, False)
    robot.setJointAnglesOfGroup('rarm', angles[3:9], tm, False)
    robot.setJointAnglesOfGroup('larm', angles[9:15], tm, True)

# def movejnts15(nxjnts):



if __name__ == "__main__":
    nxt = init()
    connect()
    #calibrateJoint()
    # servoOFF()
    # servoON()
    # goInitial()
    # setJointAngles([10,0,0,-15,0,-143,0,0,0,15,0,-143,0,0,0,0])
    # setJointAngles([-19.6,0,0,0,-25.7,-127.5,0,0,0,-17,-53.6,-91.1,-26.5,56.3,-20.3,0,0,0,-0.05])
    # setJointAngles([-19.6,0,0,0,-25.7,-127.5,0,0,0,-12.5,-40.7,-89.8,-19.9,42.1,-28.1,0,0,0,-0.05])
    # setJointAngles([-19.6,0,0,0,-25.7,-127.5,0,0,0,-24.2,-81.8,-74.7,-39.3,68.1,-39.6,0,0,0,0])
    # lhandOpen()
    # moveLArmAbs(0.5,-0.019,0.3,-180,-90,145)
    # setJointAngles([10,0,0,-15,0,-143,0,0,0,15,0,-143,0,0,0,0])
    # moveLArmAbs(0.51,-0.019,0.3,-180,-90,145)
    #goOffPose()
    


    
