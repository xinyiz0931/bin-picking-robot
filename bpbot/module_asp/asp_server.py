import os
import grpc
import time
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from concurrent import futures
import sys
import cv2
import numpy as np
import asp_pb2 as aspmsg
import asp_pb2_grpc as asprpc

class ASPServer(asprpc.ASPServicer):
    def load_aspnet(self, model_dir):
        """Load models"""
        self.model = load_model(model_dir)
        self.threshold = 0.5
        self.scores = []
        self.pred_num = -1
    
    def set_threshold(self, request, context):
        thre = request.value
        if thre >=0 and thre <=1:
            self.threshold = thre
        return aspmsg.NoValue()

    def action_grasp_inference(self, request, context):
        """module action-grasp-inference

        Args:
            probs (numpy array): shape = [grasp number, 7]

        Returns:
            action index, grasp index 
        """
        print(f"[*] Start inference, threshold = {self.threshold}!")
        probs = np.frombuffer(request.probs, dtype=np.float32)
        alist, plist = [], []
        # calculate the max decision for each grasp
        probs = np.reshape(probs, (int(len(probs)/7), 7))
        for p in probs:
            max_prob = 0 
            if (p < self.threshold).all(): 
                # no prob is larger than 0.5 ==> Action 6
                max_action = 6
                max_prob = p[6]
            elif (p >= self.threshold).all():
                max_action = 0
                max_prob = p[0]
            else: 
                for k in range(7):
                    if p[k] > self.threshold:
                        max_action = k
                        max_prob = p[k]
                        break
            alist.append(max_action)
            plist.append(max_prob)
        # calculate the final graspo index  
        for i in range(7):
            if alist.count(i) != 0:
                final_a = i
                if alist.count(i) == 1:
                    [index]=[j for j, x in enumerate(alist) if x == i]
                    final_gindex = 0
                    final_gindex = index
                    break
                elif alist.count(6)==len(alist) and (np.array(plist) < self.threshold).all()==True:
                    final_a = 6
                    # final_gindex = plist.index(max(plist))
                    final_gindex = 0 
                    break
                else:
                    indexes = [j for j,x in enumerate(alist) if x == i]
                    max_p = max([plist[j] for j in indexes])
                    
                    index=[j for j, x in enumerate(plist) if x == max_p]
                    final_gindex = index[0]
                    break
        print(f"[*] Best action: {final_a}, best grasp index: {final_gindex}")
        return aspmsg.AGPair(action=final_a, graspidx=final_gindex)
    
    def action_success_prediction(self, request, context):
        print("[*] Start prediction!  ")
        img = cv2.imread(request.imgpath)
        grasps = np.frombuffer(request.grasps)
        ch, cw, _ = img.shape
        
        g_num = int(len(grasps)/2)
        pred_num = g_num * 7
        pixel_poses = np.reshape(grasps, (g_num, 2)).astype(np.float64)
        pixel_poses[:, 0] *= (224/cw)
        pixel_poses[:, 1] *= (224/ch)
        poses = np.repeat(pixel_poses, 7, axis=0)
        sampled_actions = to_categorical(list(range(7)), 7)
        img = cv2.resize(img,(224,224))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        tileimg = cv2.cvtColor(np.tile(gray, (pred_num, 1)), cv2.COLOR_GRAY2BGR)
        images = np.reshape(tileimg, (pred_num, 224,224,3))

        actions = np.reshape(np.tile(sampled_actions, (g_num, 1)), (pred_num, 7))
        res = self.model.predict([images, poses, actions])
        self.scores = res[:, 1].reshape((g_num, 7))
        # return res[:, 1].reshape((g_num, 7))
        p2bytes = np.ndarray.tobytes(res[:, 1])
        return aspmsg.ASPOutput(probs=p2bytes)
        # return aspmsg.ASPOutput(probs = res[:, 1].reshape((g_num, 7)))
    
def serve(model_dir, host = "localhost:50051"):

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    aspserver = ASPServer()
    m = aspserver.load_aspnet(model_dir)

    asprpc.add_ASPServicer_to_server(aspserver, server)
    server.add_insecure_port(host)
    server.start()
    print("[*] The tensorflow server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    # serve(model_dir = "C:\\Users\\matsu\\Documents\\bpbot\\learning\model\\Logi_AL_20210827_145223.h5",
    #       host = "127.0.0.1:18300")
    # serve(model_dir = "/home/xinyi/Workspace/aspnet/model/Logi_AL_20210827_145223.h5",
    #       host = "localhost:50051")
    serve(model_dir = "/home/hlab/trained_models/asp_fm_20210827_145223.h5",
          host = "localhost:50051")
          
          


