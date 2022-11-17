import grpc
import numpy as np
import picksep_pb2 as psmsg
import picksep_pb2_grpc as psrpc

class PickSepClient(object):
    def __init__(self):
        try: 
            options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
            channel = grpc.insecure_channel('localhost:50050', options=options)
            self.stub = psrpc.PickSepStub(channel)
        except grpc.RpcError as rpc_error: 
            print(f"[!] Failed to connect PickSepServer: {rpc_error.code()}")
    
    def infer_picknet(self, imgpath):
        """
        Args:
            imgpath (str): path to one image

        Returns:
            pick_sep_p: [[pick_x,pick_y],[sep_x,sep_y]]
            pn_score: [pick_score,sep_score]
        """

        self.keys = ["pick_sep_p", "pn_score"]
        try: 
            out_buffer = self.stub.infer_picknet(psmsg.ImgPath(imgpath=imgpath))
            out = np.frombuffer(out_buffer.ret, dtype=float)
            return np.reshape(out[:4], (2,2)).astype(int), out[4:]

        except grpc.RpcError as rpc_error: 
            print(f"[!] Failed to connect PickNet: {rpc_error.code()}")
            return

    
    def infer_sepnet(self, imgpath):
        """
        Args:
            imgpath (str): path to one image

        Returns:
            pull_p: [x,y]
            pull_v: [x,y]

        """
        self.keys = ["pull_p", "pull_v"]
        try: 
            out_buffer = self.stub.infer_sepnet(psmsg.ImgPath(imgpath=imgpath))
            out = np.frombuffer(out_buffer.ret, dtype=float) 
            print("sepnet client: ", out)
            return out[:2].astype(int), out[2:]

        except grpc.RpcError as rpc_error:
            print(f"[!] Failed to connect SepNet: {rpc_error.code()}")
            return

if __name__ == "__main__":
    import timeit
    
    psc = PickSepClient()
    # img_path = "/home/hlab/Desktop/predicting/tmp1.png"
    img_path = "C:\\Users\\xinyi\\Desktop\\_tmp\\000132.png"
    
    start = timeit.default_timer()
    ret = psc.infer_sepnet(imgpath=img_path)
    for k, r in zip(psc.keys, ret):
        print(k, "=>", r)
    print("Time cost (SepNet):", timeit.default_timer() - start)
    
    start = timeit.default_timer()
    ret = psc.infer_picknet(imgpath=img_path)
    for k, r in zip(psc.keys, ret):
        print(k, "=>", r)
    print("Time cost (PickNet): ", timeit.default_timer() - start)
    