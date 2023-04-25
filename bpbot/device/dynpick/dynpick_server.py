import os
import time
from concurrent import futures
import numpy as np
import warnings
warnings.filterwarnings("ignore")

import grpc
import dynpick_pb2 as dpmsg
import dynpick_pb2_grpc as dprpc

class DynPickServer(dprpc.DynPickServicer):
    def __init__(self):
        from bpbot.device import DynPickControl
        self.ft = DynPickControl()
        self.fdc = self.ft.connect()
    
    def get_raw(self, request, context):
        clkb = 0
        while True:
            clk = (time.process_time()) * 1000
            if clk >= (clkb + self.ft.tw):
                clkb = clk/self.ft.tw * self.ft.tw
                os.write(self.fdc, str.encode('R'))
                l = os.read(self.fdc, 27)
                if l != bytes():
                    break
        return l
    
    def get(self, request, context):
        # clkb = 0
        # while True:
        #     clk = (time.process_time()) * 1000
        #     if clk >= (clkb + self.ft.tw):
        #         clkb = clk/self.ft.tw * self.ft.tw
        #         os.write(self.fdc, str.encode('R'))
        #         l = os.read(self.fdc, 27)
        #         if l != bytes():
        #             break
        l = self.get_raw(request,context)

        detected_load = []
        for i in range(1,22,4):
            detected_load.append(int((l[i:i+4]).decode(),16))
        # detected_load[:3] = (detected_load[:3]-self.ft.zero_load[:3])/self.ft.force_sensitivity
        # detected_load[3:] = (detected_load[3:]-self.ft.zero_load[3:])/self.ft.torque_sensitivity
        l2bytes = np.ndarray.tobytes(np.array(detected_load))
        return dpmsg.Data(data=l2bytes)
    
    # def save(self, request, context):
    #     open(self.ft.out_path, 'w').close()

    #     def write():
    #         t = 1
    #         while True:
    #             line = [t*self.ft.tw]
    #             l = self.get_raw(request, context)
    #             for i in range(1,22,4):
    #                 line.append(int((l[i:i+4]).decode(),16))
    #             with open(self.ft.out_path, 'a') as fp:
    #                 print(*line, file=fp)
    #                 t += 1
    #             time.sleep(self.ft.tw/1000)

    #     self.p = Process(target=write)
    #     self.p.start()
        
    #     return dpmsg.Nothing()
    
    # def save_ok(self, request, context):
    #     if self.p is not None and self.p.is_alive():
    #         self.p.terminate()

    #     return dpmsg.Nothing()

def serve(host="localhost:2222"):
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    psserver = DynPickServer()
    dprpc.add_DynPickServicer_to_server(psserver, server)
    server.add_insecure_port(host)
    server.start()
    print("[*] Force sensor server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0) 

if __name__ == '__main__':
    serve()
