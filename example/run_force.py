from multiprocessing import Process
import time
from bpbot.device import FTSensor
import numpy as np
def run2(s, n):
    for i in range(5):
        print("----------------",i,"--------------------")
        for i in range(n):
            print(str(n-i)+'s')
            time.sleep(1)
        with open("/home/hlab/bpbot/data/force/out.txt", 'a') as fp:
            print(*([-1]*7), file=fp)
    t1.terminate()

def record():
    import os
    os.system("bash /home/hlab/bpbot/script/start_ft.sh")

sensor = FTSensor()
# t1 = Process(target=sensor.record, args=())
t1 = Process(target=record, args=())
t2 = Process(target=run2, args=('t2',3))
t1.start()
t2.start()
t1.join()
t2.join()
