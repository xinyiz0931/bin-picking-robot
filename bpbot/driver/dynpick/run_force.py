import os
import sys
import time
import glob
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import configparser
import termios
import numpy as np



# find port of sensor and open
try:
    port = (glob.glob(r'/dev/ttyUSB*'))[0]
    # os.system("sudo chmod a+rw " + port)
    fdc = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    print("Open port "+port)
except BlockingIOError as e:
    print("Can't open port! ")
    fdc = -1

if (fdc < 0):
    os.close(fdc)

############# tty control ################
term_ = termios.tcgetattr(fdc)
term_[0] = termios.IGNPAR #iflag
term_[1] = 0 # oflag
term_[2] = termios.B921600 | termios.CS8 | termios.CLOCAL | termios.CREAD # cflag
term_[3] = 0 # lflag -> ICANON
term_[4] = 4103 # ispeed
term_[5] = 4103 # ospeed
# # cc
o_ = bytes([0])
term_[6][termios.VINTR] = o_ # Ctrl-c
term_[6][termios.VQUIT] = o_ # Ctrl-?
term_[6][termios.VERASE] = o_ # del
term_[6][termios.VKILL] = o_ # @
term_[6][termios.VEOF] =  bytes([4])# Ctrl-d
term_[6][termios.VTIME] = 0
term_[6][termios.VMIN] = 0
term_[6][termios.VSWTC] = o_ # ?0
term_[6][termios.VSTART] = o_ # Ctrl-q
term_[6][termios.VSTOP] = o_ # Ctrl-s
term_[6][termios.VSUSP] = o_ # Ctrl-z
term_[6][termios.VEOF] = o_ # ?0
term_[6][termios.VREPRINT] = o_ # Ctrl-r
term_[6][termios.VDISCARD] = o_ # Ctrl-u
term_[6][termios.VWERASE] = o_ # Ctrl-w
term_[6][termios.VLNEXT] = o_ # Ctrl-v
term_[6][termios.VEOL2] = o_ # ?0

termios.tcsetattr(fdc, termios.TCSANOW, term_)
################## over ##################

tw = 50
clkb = 0
clkb2 = 0
num = 0
clk0 = (time.process_time())*1000 # ms

r_ = str.encode("R")
os.write(fdc, r_)

# initialization
fp = open('./out.txt','wt')
fig = plt.figure(figsize=(12,5))

ax = fig.add_subplot(1, 2, 1)
ax.grid()
ax3d = fig.add_subplot(1,2,2, projection='3d')
# plt.ylim([8,10]) # not adjust
x0 = 8167 * 0.001
y0 = 8247 * 0.001
z0 = 8541 * 0.001
fline = [0, x0, y0, z0] # initial values

def vector_rotate(x,y,theta):
    """Only for 2D"""
    x_ = y*np.sin(theta)+x*np.cos(theta)
    y_ = y*np.cos(theta)+x*np.sin(theta)
    return x_, y_

def sensor_to_robot(s_x,s_y,s_z):

    angle = 0
    if int(angle) == 0:
        r_x, r_y = s_x, -s_y
    elif int(angle) == 45:
        theta = 0.25*np.pi
        s_x_, s_y_ = vector_rotate(s_x, s_y, theta)
        r_x, r_y = s_x, -s_y
    elif int(angle) == 90:
        r_x, r_y = s_y, s_x
    else:
        theta = 0.25*np.pi
        s_x_, s_y_ = vector_rotate(s_x, s_y, theta)
        r_x, r_y = -s_y, -s_x
    # theta = (angle/180)*np.pi # defined by gripper angle
    # r_x, r_y = vector_rotate(s_x, s_y, theta)
    return (-r_x, -r_y, -s_z)

# start recording
while True:
    try:
        data = []
        while True:
            clk = (time.process_time()) * 1000 - clk0
            if clk >= (clkb + tw):
                clkb = clk / tw * tw
                break
        os.write(fdc, r_)
        l = os.read(fdc, 27)
        data.append(str(int(clk / tw * tw)))
        for i in range(1,22,4):
            # bytes -> string(for file writting) -> decimal(for vis)
            data.append(str(int((l[i:i+4]).decode(),16)))
        if len(l) < 27:
            print("error! quit! ")
        fp.write(",".join(data))
        fp.write("\n")
        
        # try visualization
        # plot [time, fx]
        point1 = [int(data[0])*0.001, int(data[1])*0.001]
        point2 = [fline[0], fline[1]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        ax.plot(x_values, y_values, '-g',alpha=0.3)

        # plot [time, fy]
        point1 = [int(data[0])*0.001, int(data[2])*0.001]
        point2 = [fline[0], fline[2]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]

        ax.plot(x_values, y_values, '-b',alpha=0.3)

        # plot [time, fz]
        point1 = [int(data[0])*0.001, int(data[3])*0.001]
        point2 = [fline[0], fline[3]]
        x_values = [point1[0], point2[0]]
        y_values = [point1[1], point2[1]]
        ax.plot(x_values, y_values, '-r')

        x_use = (fline[1]+int(data[1])*0.001)/2
        y_use = (fline[2]+int(data[2])*0.001)/2
        z_use = (fline[3]+int(data[3])*0.001)/2

        fline[0]=int(data[0])*0.001#time
        fline[1]=int(data[1])*0.001#fx
        fline[2]=int(data[2])*0.001#fy
        fline[3]=int(data[3])*0.001#fz

        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Force (N)')

        # fig.canvas.draw()
        # # try compute vector
        # ax = fig.add_subplot(2, 1, 2, projection='3d')
        # x_, y_, z_ = ef.sensor_to_robot(fline[1]-x0,fline[2]-y0,fline[3]-z0) 
        x_, y_, z_ = sensor_to_robot(x_use-x0,y_use-y0,z_use-z0) 
        print("The z value is {}".format(np.abs(z_)))
        ax3d.cla()
        ax3d.quiver(0,0,0,x_,y_, z_, alpha=0.5,arrow_length_ratio=0.1, color='red')
        
        # ax3d.set_xlim([-0.7,0.7])
        # ax3d.set_ylim([-0.7,0.7])
        ax3d.set_zlim([0.3,-0.3])
        ax3d.set_xlabel('x axis',color="g")
        ax3d.set_ylabel('y axis',color="b")
        ax3d.set_zlabel('z axis',color="r")
        ax3d.view_init(-163,128)
        plt.pause(0.001)

        # detect key press

    except KeyboardInterrupt:
        break








