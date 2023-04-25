#!/usr/bin/env python
import copy
import math
import numpy
import os
import time
import socket

import roslib
roslib.load_manifest("hrpsys")
from hrpsys.hrpsys_config import *

# hot fix for https://github.com/start-jsk/rtmros_hironx/issues/539
#  On 16.04 (omniorb4-dev 4.1) if we start openhrp-model-loader with
# <env name="ORBgiopMaxMsgSize" value="2147483648" />
# all connection(?) conect respenct giopMaxMsgSize
#  But on 18.04 (omniorb4-dev 4.2) wnneed to set ORBgiopMaxMsgSize=2147483648
# for each clients
if not os.environ.has_key('ORBgiopMaxMsgSize'):
    os.environ['ORBgiopMaxMsgSize'] = '2147483648'

import OpenHRP
import OpenRTM_aist
import OpenRTM_aist.RTM_IDL
from hrpsys import rtm
#import rtm
#from rtm import *
from OpenHRP import *
import waitInput
from waitInput import *

from distutils.version import StrictVersion

SWITCH_ON = OpenHRP.RobotHardwareService.SWITCH_ON
SWITCH_OFF = OpenHRP.RobotHardwareService.SWITCH_OFF
_MSG_ASK_ISSUEREPORT = 'Your report to ' + \
                       'https://github.com/start-jsk/rtmros_hironx/issues ' + \
                       'about the issue you are seeing is appreciated.'
_MSG_RESTART_QNX = 'You may want to restart QNX/ControllerBox afterward'

from distutils.version import StrictVersion

def arrayDistance (angle1, angle2):
    return sum([abs(i-j) for (i,j) in zip(angle1,angle2)])

def arrayApproxEqual (angle1, angle2, thre=1e-3):
    return arrayDistance(angle1, angle2) < thre

def saveLogForCheckParameter(log_fname="/tmp/test-samplerobot-emergency-stopper-check-param"):
    hcf.setMaxLogLength(1);hcf.clearLog();time.sleep(0.1);hcf.saveLog(log_fname)

def checkParameterFromLog(port_name, log_fname="/tmp/test-samplerobot-emergency-stopper-check-param", save_log=True, rtc_name="es"):
    if save_log:
        saveLogForCheckParameter(log_fname)
    return map(float, open(log_fname+"."+rtc_name+"_"+port_name, "r").readline().split(" ")[1:-1])

def getWrenchArray ():
    saveLogForCheckParameter()
    return reduce(lambda x,y: x+y, (map(lambda fs : checkParameterFromLog(fs+"Out", save_log=False), ['lfsensor', 'rfsensor', 'lhsensor', 'rhsensor'])))

def init():
    robotname = "HiroNX(Robot)0"
    url = ""
    # ---------------- for real robot --------------------
    robotname = "RobotHardware0"
    host = '192.168.128.10'
    port = '15005'
    print('host:' + host)
    print('port:' + port)
    rtm.nshost = host
    rtm.nsport = port
    # ----------------------------------------------------

    global hcf, init_pose, reset_pose, hrpsys_version
    hcf = HrpsysConfigurator()
    _init_pose = [0, 0, 0, -10.0003, -25.7008, -127.504, 0, 0, 0, 23.0007, -25.7008, -127.504, -7.0002, 0, 0]
    init_pose = [x / 180.0 * math.pi for x in _init_pose] 
    _reset_pose = [32.957, 0, 0, 8.96503, -45.1829, -69.629, 16.559, 26.2259, -34.7475, 23.0007, -25.7008, -127.504, -7.0002, 0, 0]
    reset_pose = [x / 180.0 * math.pi for x in _reset_pose] 
    
    print hcf.getRTCList()
    #hcf.getRTCList = hcf.getRTCListUnstable
    
    hcf.init(robotname=robotname, url=url)
    hrpsys_version = hcf.fk.ref.get_component_profile().version

def demoEmergencyStopJointAngle():
    print >> sys.stderr, "1. test stopMotion and releaseMotion for joint angle"
    hcf.es_svc.releaseMotion()
    hcf.seq_svc.setJointAngles(init_pose, 1.0)
    hcf.waitInterpolation()
    time.sleep(0.1)
    tmp_angle1 = hcf.getActualState().angle
    play_time = 10
    hcf.seq_svc.setJointAngles(reset_pose, play_time)
    print >> sys.stderr, "  send angle_vector of %d [sec]" % play_time
    time.sleep(4)
    print >> sys.stderr, "  check whether robot pose is changing"
    tmp_angle2 = hcf.getActualState().angle
    if arrayApproxEqual(init_pose, tmp_angle1) and not(arrayApproxEqual(tmp_angle1, tmp_angle2)):
        print >> sys.stderr, "  => robot is moving."
    assert (arrayApproxEqual(init_pose, tmp_angle1) and not(arrayApproxEqual(tmp_angle1, tmp_angle2)))
    print >> sys.stderr, "  stop motion"
    hcf.es_svc.stopMotion()
    time.sleep(0.1)
    print >> sys.stderr, "  check whether robot pose remained still"
    tmp_angle1 = hcf.getActualState().angle
    time.sleep(3)
    tmp_angle2 = hcf.getActualState().angle
    if arrayApproxEqual(tmp_angle1, tmp_angle2):
        print >> sys.stderr, "  => robot is not moving. stopMotion is working succesfully."
    assert (arrayApproxEqual(tmp_angle1, tmp_angle2))
    print >> sys.stderr, "  release motion"
    hcf.es_svc.releaseMotion()
    print >> sys.stderr, "  check whether robot pose changed"
    tmp_angle1 = hcf.getActualState().angle
    hcf.waitInterpolation()
    time.sleep(0.1)
    tmp_angle2 = hcf.getActualState().angle
    if (not(arrayApproxEqual(tmp_angle1, tmp_angle2)) and arrayApproxEqual(tmp_angle2, reset_pose)):
        print >> sys.stderr, "  => robot is moving. releaseMotion is working succesfully."
    assert(not(arrayApproxEqual(tmp_angle1, tmp_angle2)) and arrayApproxEqual(tmp_angle2, reset_pose))
    hcf.es_svc.releaseMotion()
    hcf.seq_svc.setJointAngles(init_pose, 1.0)
    hcf.waitInterpolation()

def demo(key_interaction=False):
    init()
    from distutils.version import StrictVersion
    if hcf.es != None:
        print "EmergencyStopper found"
    else: 
        print "EmergencyStopper not found"
    
    if StrictVersion(hrpsys_version) >= StrictVersion('315.6.0'):
        if hcf.es != None:
            demoEmergencyStopJointAngle()
    else: 
        print "hrpsys version is lower than 315.6.0"

if __name__ == '__main__':
    demo()


