from termcolor import colored, cprint
import numpy as np
import math
from scipy.spatial.transform import Rotation as R

"""
All quaternions are formatted as `(x,y,z,w)`
Pay attention with the phycics simulator
"""
###########################################################################
################################# MATH UTILS ##############################
###########################################################################

def quaternion2mat(Rq):
    r = R.from_quat(Rq)
    return r.as_matrix()

def quaternion2rpy(Rq):
    # unit: degree
    r = R.from_quat(Rq)
    return r.as_euler('xyz', degrees=True)

def rpy2quaternion(Re):
    r= R.from_euler('xyz', Re, degrees=True)
    return r.as_quat()
    
def rotate_3d(p, R, origin=(0, 0, 0)):
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T - o.T) + o.T).T)

###########################################################################
############################ PRINT UTILS ##################################
###########################################################################
def main_proc_print(main_proc_str):
    (lambda x: cprint(x, 'yellow'))("[ MAIN PROCESS ] "+str(main_proc_str))
    

def warning_print(warning_str):
    (lambda x: cprint(x, 'red'))("[   WARNINGS   ] "+str(warning_str))

def result_print(result_str):
    (lambda x: cprint(x, 'green'))("[    OUTPUT    ] "+str(result_str))


def important_print(result_str):
    (lambda x: cprint(x, 'white',  'on_green'))("[    NOTICE    ] "+str(result_str))
