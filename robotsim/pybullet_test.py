import os
import pybullet as p
import pybullet_data
import math

p.connect(p.GUI)
urdfRootPath=pybullet_data.getDataPath()
pandaUid = p.loadURDF(os.path.join(urdfRootPath, "franka_panda/panda.urdf"),useFixedBase=True)

tableUid = p.loadURDF(os.path.join(urdfRootPath, "table/table.urdf"),basePosition=[0.5,0,-0.65])

trayUid = p.loadURDF(os.path.join(urdfRootPath, "tray/traybox.urdf"),basePosition=[0.65,0,0])

p.setGravity(0,0,-10)
objectUid = p.loadURDF(os.path.join(urdfRootPath, "random_urdfs/000/000.urdf"), basePosition=[0.7,0,0.1])

p.resetDebugVisualizerCamera(cameraDistance=1.5, cameraYaw=0, cameraPitch=-40, cameraTargetPosition=[0.55,-0.35,0.2])

state_durations = [1,1,1,1]
control_dt = 1./240.
p.setTimestep = control_dt
state_t = 0.
current_state = 0

while True:
    state_t += control_dt
    p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING) 
    if current_state == 0:
        p.setJointMotorControl2(pandaUid, 0, 
                        p.POSITION_CONTROL,0)
        p.setJointMotorControl2(pandaUid, 1, 
                        p.POSITION_CONTROL,math.pi/4.)
        p.setJointMotorControl2(pandaUid, 2, 
                        p.POSITION_CONTROL,0)
        p.setJointMotorControl2(pandaUid, 3, 
                        p.POSITION_CONTROL,-math.pi/2.)
        p.setJointMotorControl2(pandaUid, 4, 
                        p.POSITION_CONTROL,0)
        p.setJointMotorControl2(pandaUid, 5, 
                        p.POSITION_CONTROL,3*math.pi/4)
        p.setJointMotorControl2(pandaUid, 6, 
                        p.POSITION_CONTROL,-math.pi/4.)
        p.setJointMotorControl2(pandaUid, 9, 
                        p.POSITION_CONTROL, 0.08)
        p.setJointMotorControl2(pandaUid, 10, 
                        p.POSITION_CONTROL, 0.08)
    if current_state == 1:
        p.setJointMotorControl2(pandaUid, 1, 
                        p.POSITION_CONTROL,math.pi/4.+.15)
        p.setJointMotorControl2(pandaUid, 3, 
                        p.POSITION_CONTROL,-math.pi/2.+.15)
    if current_state == 2:
        p.setJointMotorControl2(pandaUid, 9, 
                        p.POSITION_CONTROL, 0.0, force = 200)
        p.setJointMotorControl2(pandaUid, 10, 
                        p.POSITION_CONTROL, 0.0, force = 200)
    if current_state == 3:
        p.setJointMotorControl2(pandaUid, 1, 
                        p.POSITION_CONTROL,math.pi/4.-1)
        p.setJointMotorControl2(pandaUid, 3, 
                        p.POSITION_CONTROL,-math.pi/2.-1)

    if state_t >state_durations[current_state]:
        current_state += 1
        if current_state >= len(state_durations):
            current_state = 0
        state_t = 0
    p.stepSimulation()