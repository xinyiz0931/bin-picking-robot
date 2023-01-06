from multiprocessing import Process
import time
from bpbot.device import FTSensor
def run2(s, n):
    for i in range(n):
        print(str(n-i)+'s')
        time.sleep(1)
    t1.terminate()

def run1():
    print("main")

sensor = FTSensor()
# t1 = Process(target=sensor.record, args=())
t1 = Process(target=run1, args=())
t2 = Process(target=run2, args=('t2',5))
t1.start()
t2.start()
t1.join()
t2.join()






