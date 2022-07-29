import os
import time
import grpc
from concurrent import futures
from tangle import Config, Inference
from tangle.utils import direction2vector 
import numpy as np
import picksep_pb2 as psmsg
import picksep_pb2_grpc as psrpc

class PickSepServer(psrpc.PickSepServicer): 
    def initialize(self):        
        # create config for the server

        config_data = {
            # "root_dir": "D:\\",
            "root_dir": "/home/hlab/Documents/",
            "infer": 
            {
                "infer_type": "pick_sep",
                "mode": "test",
                "use_cuda": True,
                "batch_size": 1,
                "img_height": 512,
                "img_width": 512,
                # "pick_ckpt_folder": ["try_retrain_picknet_unet","model_epoch_15.pth"],
                "pick_ckpt_folder": ["try8","model_epoch_8.pth"],
                "sepp_ckpt_folder": ["try_38","model_epoch_3.pth"],
                "sepd_ckpt_folder": ["try_new_res","model_epoch_11.pth"]
                # "sepd_ckpt_folder": ["try_new","model_epoch_11.pth"]
            }
        }
        # config_path = os.path.join(tangle.__file__, "../../cfg/config.yaml")
        
        cfg = Config(config_type="infer", config_data=config_data)
        self.save_dir = "/home/hlab/bpbot/data/depth/"
        self.inference = Inference(config=cfg)
    
    def infer_picknet(self, request, context): 
        """
        Returns:
            0 or 1: classification of pick or sep
            array (4,): [u,v,score_pick, score_sep]
        """
        print(f"[*] Start inference: picknet")
        outputs = self.inference.infer(data_dir=request.imgpath, infer_type="pick", save_dir=self.save_dir)
        pickorsep = int(outputs[0][0])
        scores = outputs[2][0]
        
        # pick
        if pickorsep == 0:
            action = np.array([outputs[1][0][0], outputs[2][0]])
        # sep
        else:
            action = np.array([outputs[1][0][1], outputs[2][0]])
        
        # if pickorsep: grasps = outputs[1][0][1] # sep
        # else: grasps = outputs[1][0][0] # pick 
        print(pickorsep,action)
        p2bytes = np.ndarray.tobytes(np.array(action))
        return psmsg.ActionCls(pickorsep=pickorsep, action=p2bytes)
    
    def infer_picknet_sepnet_pos(self, request, context):
        print(f"[*] Start inference: picknet + sepnet-p! ")
        outputs = self.inference.infer(data_dir=request.imgpath, infer_type="pick_sep_pos", save_dir=self.save_dir)
        pickorsep = int(outputs[0][0])
        # sep
        if pickorsep: 
            action = np.array([outputs[2][0][0], outputs[2][0][1]])
        else: 
            action = np.array(outputs[1][0][0])

        return psmsg.ActionCls(pickorsep=pickorsep, action=np.ndarray.tobytes(action)) 
        
    def infer_picknet_sepnet(self, request, context):
        print(f"[*] Start inference: pickenet + sepnet")
        outputs = self.inference.infer(data_dir=request.imgpath, infer_type="pick_sep", save_dir=self.save_dir)
        pickorsep = int(outputs[0][0])
        if pickorsep: 
            scores = np.array(outputs[4][0])
            max_direction = scores.argmax()
            # max_score = scores.max()
            pull_v = direction2vector(max_direction*360/16)
            action = np.array([outputs[2][0][0], outputs[2][0][1], pull_v, outputs[3][0]])
        else: 
            action = np.array([outputs[1][0][0], outputs[3][0]]) # pick 
        return psmsg.ActionCls(pickorsep=pickorsep, action=np.ndarray.tobytes(action)) 

def serve(host="localhost:50050"):
    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    psserver = PickSepServer()
    m = psserver.initialize()
    psrpc.add_PickSepServicer_to_server(psserver, server)
    server.add_insecure_port(host)
    server.start()
    print("[*] PickNet + SepNet server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0) 

if __name__ == "__main__":
    serve(host="localhost:50050")