import os
import cv2
import grpc
import time
import numpy as np

import sys
import asp_pb2 as aspmsg
import asp_pb2_grpc as asprpc

class ASPClient(object):
    def __init__(self):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel('localhost:50051', options=options)
        self.stub = asprpc.ASPStub(channel)

    def set_threshold(self, threshold):
        self.stub.set_threshold(aspmsg.DValue(value=threshold))

    def predict(self, imgpath, grasps):
        try:
            g2bytes = np.ndarray.tobytes(grasps)
            #g2bytes = grasps
            out = self.stub.action_success_prediction(aspmsg.ASPInput(imgpath=imgpath, grasps=g2bytes))
            return np.frombuffer(out.probs, dtype=np.float32)
        except grpc.RpcError as rpc_error:
            print(f"[!] Failed with {rpc_error.code()}")
            return

    def infer(self,imgpath, grasps):
        try:
            g2bytes = np.ndarray.tobytes(grasps)             
            out = self.stub.action_success_prediction(aspmsg.ASPInput(imgpath=imgpath, grasps=g2bytes))
            res = self.stub.action_grasp_inference(out)
            return res.action, res.graspidx
        except grpc.RpcError as rpc_error:
            print(f'Failed with {rpc_error.code()}')
            return

if __name__ == "__main__":


    import timeit
    start = timeit.default_timer()

    
    import bpbot.module_asp.asp_client as aspclt
    aspc = aspclt.ASPClient()
    aspc.set_threshold(0.4)
    imgpath = "/home/hlab/bpbot/data/test/depth3.png"
    g = np.array([[4, 5], [6, 7]])
    
    # g2bytes=np.ndarray.tobytes(g)
    res_p = aspc.predict(imgpath=imgpath, grasps=g)
    # print(res_p)

    a,g = aspc.infer(imgpath=imgpath, grasps=g)
    print(a,g)
        # print(res.action, res.graspno)
    #end = timeit.default_timer()
    #print("Time cost: ", end-start)
