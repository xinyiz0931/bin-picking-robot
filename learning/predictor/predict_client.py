import os
import grpc
import time
import numpy as np

import sys
sys.path.append("./")

import learning.predictor.predictor_pb2 as pdmsg
import learning.predictor.predictor_pb2_grpc as pdrpc

class PredictorClient(object):
    def __init__(self):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel('localhost:50051', options=options)
        self.stub = pdrpc.PredictorStub(channel)

    def predict(self,imgpath, grasps):
        return self.stub.predict(pdmsg.Grasp(imgpath=imgpath,grasps=grasps))

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    import predict_client as pdclt
    pdc = pdclt.PredictorClient()
    imgpath = "/home/xinyi/Workspace/myrobot/vision/depth/depth.png"
    y = np.array([[4, 5], [6, 7]])
    y2bytes=np.ndarray.tobytes(y)
    pdc.predict(imgpath=imgpath, grasps=y2bytes)

    end = timeit.default_timer()
    print("Time cost: ", end-start)
    # pxc.saveply("../../vision/pointcloud/out.ply")
    # print("point cloud is captured and saved! ")