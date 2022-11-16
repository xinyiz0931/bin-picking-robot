import grpc
import numpy as np
import picksep_pb2 as psmsg
import picksep_pb2_grpc as psrpc

class PickSepClient(object):
    def __init__(self):
        options = [('grpc.max_receive_message_length', 100 * 1024 * 1024)]
        channel = grpc.insecure_channel('localhost:50050', options=options)
        self.stub = psrpc.PickSepStub(channel)

    def infer_picknet(self,imgpath):
        """PickNet

        Args:
            imgpath (str): path to image

        Returns:
            pick_sep_points: shape=(2,2)
            pick_sep_score: shape=(2,)
        """
        try: 
            out_buffer = self.stub.infer_picknet(psmsg.ImgPath(imgpath=imgpath))
            out = np.frombuffer(out_buffer.ret, dtype=float)
            return np.reshape(out[:4], (2,2),).astype(int), out[4:] 
        except grpc.RpcError as rpc_error: 
            print(f"[!] PickNet failed with {rpc_error.code()}")
            return

    def infer_sepnet(self, imgpath):
        """SepNet

        Args:
            imgpath (str): path ot one image

        Returns:
            pull_point: shape=(2,)
            pull_vector: shape=(2,)
        """
        try: 
            out_buffer = self.stub.infer_sepnet(psmsg.ImgPath(imgpath=imgpath))
            out = np.frombuffer(out_buffer.ret, dtype=float) 
            return out[:2].astype(int), out[2:]

        except grpc.RpcError as rpc_error:
            print(f"[!] SepNet failed with {rpc_error.code()}")
            return


    # def infer_picknet(self, imgpath):
    #     """
    #     Args:
    #         imgpath (str): path to one image

    #     Returns:
    #         pickorsep: 0->pick/1->sep
    #         pick_sep_p: [x,y]
    #         pn_score: [s_pick,s_sep]
    #     """
    #     try: 
    #         out_buffer = self.stub.infer_picknet(psmsg.ImgPath(imgpath=imgpath))
    #     except grpc.RpcError as rpc_error: 
    #         print(f"[!] PickNet failed with {rpc_error.code()}")
    #         return
    #     out = np.frombuffer(out_buffer.ret, dtype=float)
    #     return out[0].astype(int), out[1:3].astype(int), out[3:]
    
    # def infer_sepnet(self, imgpath):
    #     """
    #     Args:
    #         imgpath (str): path to one image

    #     Returns:
    #         pull_hold_p: [[x,y],[x,y]]
    #         pull_v: [x,y]
    #         snd_score: [s] * # directions
    #     """
    #     try: 
    #         out_buffer = self.stub.infer_sepnet(psmsg.ImgPath(imgpath=imgpath))
            
    #         out = np.frombuffer(out_buffer.ret, dtype=float) 
    #         return np.reshape(out[:4], (2,2),).astype(int), out[4:6], out[6:]

    #     except grpc.RpcError as rpc_error:
    #         print(f"[!] SepNet failed with {rpc_error.code()}")
    #         return

    # def infer_picknet_sepnet(self, imgpath, sep_motion):
        """
        Args:
            imgpath (str): path to one image
            sep_motion (bool): separation motions is needed?  

        Returns:
            if sep_motion: 
                case 1. 0->pick => pickorsep: 0
                                pick_sep_p: [x,y]
                                pn_score: [s_pick,s_sep] 
                case 2. 1->sep  => pickorsep: 1
                                pn_score: [s_pick,s_sep] 
                                pull_hold_p: [[x,y],[x,y]]
                                pull_v: [x,y]
                                snd_score: [s] * # directions
            elif not sep_motion: 
                pickorsep: 0->pick/1->sep
                pick_sep_p: [x,y]
                pn_score: [s_pick,s_sep] 
        """
        
        pn_out = self.infer_picknet(imgpath)
        if not pn_out:  
            print("[!] PickNet + SepNet failed! ")
            return
        if sep_motion and pn_out[0] == 1:
            sn_out = self.infer_sepnet(imgpath)
            return pn_out[0], pn_out[-1], *sn_out
        else:
            return pn_out

if __name__ == "__main__":
    import timeit
    start = timeit.default_timer()
    
    psc = PickSepClient()
    img_path = "C:\\Users\\xinyi\\Desktop\\_tmp\\000132.png"
    # img_path = "C:\\Users\\xinyi\\Documents\\XYBin_Pick\\bin\\tmp\\depth.png"
    # img_path = "C:\\Users\\xinyi\\Documents\\Dataset\\SepDataAllPullVectorEight\\images\\000394.png"
    # img_path = "D:\\Code\\bpbot\\data\\test\\depth20.png"
    



    points, scores = psc.infer_picknet(imgpath=img_path)
    pull_point, pull_vector = psc.infer_sepnet(imgpath=img_path)
    # print(p.shape, p)
    # print(s.shape, s)






    # res = psc.infer_sepnet(imgpath=img_path)
    # # res = psc.infer_picknet(imgpath=img_path)
    # res = psc.infer_picknet_sepnet(imgpath=img_path, sep_motion=True)
    # x, y  = res[1]

    # for i, r in enumerate(res):
    #     print(i, "=>", r)
    
    # print("Time cost: ", timeit.default_timer() - start)