import grpc
import numpy as np
import picksep_pb2 as psmsg
import picksep_pb2_grpc as psrpc

class PickSepClient(object):
    def __init__(self):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel('localhost:50050', options=options)
        self.stub = psrpc.PickSepStub(channel)
    
    def infer_picknet(self, imgpath):
        try: 
            outputs = self.stub.infer_picknet(psmsg.ImgPath(imgpath=imgpath))
            return outputs.pickorsep, np.frombuffer(outputs.action, dtype=float)
        except grpc.RpcError as rpc_error: 
            print(f"Failed with {rpc_error.code()}")
            return

    def infer_picknet_sepnet(self, imgpath):
        try:
            outputs = self.stub.infer_picknet_sepnet(psmsg.ImgPath(imgpath=imgpath))
            return outputs.pickorsep, np.frombuffer(outputs.action, dtype=float)
        except grpc.RpcError as rpc_error: 
            print(f"Failed with {rpc_error.code()}")
            return

    def infer_picknet_sepnet_pos(self, imgpath):

        outputs = self.stub.infer_picknet_sepnet_pos(psmsg.ImgPath(imgpath=imgpath))
        return outputs.pickorsep, np.frombuffer(outputs.action, dtype=np.int0)

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    
    psc = PickSepClient()
    # img_path = "D:\\Dataset\\sepnet\\test\\depth10.png"
    img_path = "/home/hlab/bpbot/data/test/depth3.png"
    cls, grasp = psc.infer_picknet(imgpath=img_path)
    # cls, grasp = psc.infer_picknet_sepnet(imgpath=img_path)
    
    print(cls, grasp)
    print("Time cost: ", timeit.default_timer() - start)