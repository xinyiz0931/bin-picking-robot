import os
import grpc
import time
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model

from concurrent import futures
import sys
import numpy as np
if '/opt/ros/kinetic/lib/python2.7/dist-packages' in sys.path:
        sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import cv2
import predictor_pb2 as pdmsg
import predictor_pb2_grpc as pdrpc

class PredictorServer(pdrpc.PredictorServicer):
    def load_aspnet(self, model_dir):
        """Load models"""
        self.model = load_model(model_dir)
    
    def action_selection_policy(self, prob):
        final_prob =0
        if (prob < 0.5).all(): 
            # no prob is larger than 0.5 ==> Action 6
            char = 6
            final_prob = prob[6]
        elif (prob >= 0.5).all():
            char = 0
            final_prob = prob[0]
        else: 
            for k in range(7):
                if prob[k] > 0.5:
                    char = k
                    final_prob = prob[k]
                    break
        return char, final_prob

    def predict(self,request,context):
        print("Start prediction!  ")

        img = cv2.imread(request.imgpath)
        grasps = np.frombuffer(request.grasps)
        ch, cw, _ = img.shape

        g_num = int(len(grasps)/5)
        grasps = np.reshape(grasps, (g_num, 5))
        pred_num = g_num * 7
        
        extract_poses = np.array(grasps, dtype=np.float64)[:,1:3]
        
        extract_poses[:,0] *= (224/cw)
        extract_poses[:,1] *= (224/ch)

        poses = np.repeat(extract_poses, 7, axis=0)

        sampled_actions = to_categorical(list(range(7)), 7)
        img = cv2.resize(img,(224,224))
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        tileimg = cv2.cvtColor(np.tile(gray, (pred_num, 1)), cv2.COLOR_GRAY2BGR)
        images = np.reshape(tileimg, (pred_num, 224,224,3))

        # tileimg = np.tile(gray, (pred_num, 1))
        # images = np.reshape(tileimg, (pred_num, 224,224,1))

        actions = np.reshape(np.tile(sampled_actions, (g_num, 1)), (pred_num, 7))
        # poses = np.reshape(np.tile(extract_poses, (7, 1)), (pred_num, 2))
        # actions = to_categorical((np.repeat(list(range(7)),g_num)), 7)
        res = self.model.predict([images, poses, actions])

        alist, plist = [], []
        for i in range(0,pred_num, 7):
            # in the order of the input grasps
            prob = (res[i:(i+7)])[:,1]
            a, p = self.action_selection_policy(prob)
            alist.append(a)
            plist.append(p)

        # action_indexes = sorted(range(len(alist)), key=lambda k: alist[k])
        for i in range(7):
            if alist.count(i) != 0:
                final_a = i
                if alist.count(i) == 1:
                    [index]=[j for j, x in enumerate(alist) if x == i]
                    final_gindex = 0
                    final_gindex = index
                elif alist.count(6)==len(alist) and (np.array(plist) < 0.5).all()==True:
                    final_a = 6
                    final_gindex = 0
                else:
                    indexes = [j for j,x in enumerate(alist) if x == i]
                    max_p = max([plist[j] for j in indexes])
                    
                    index=[j for j, x in enumerate(plist) if x == max_p]
                    final_gindex = index[0]

        return pdmsg.Action(action=final_a, graspno=final_gindex)

def serve(model_dir, host = "localhost:50051"):

    _ONE_DAY_IN_SECONDS = 60 * 60 * 24
    options = [('grpc.max_message_length', 100 * 1024 * 1024)]
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), options = options)
    pdserver = PredictorServer()
    m = pdserver.load_aspnet(model_dir)

    pdrpc.add_PredictorServicer_to_server(pdserver, server)
    server.add_insecure_port(host)
    server.start()
    print("The tensorflow server is started!")
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == "__main__":
    # serve(model_dir = "C:\\Users\\matsu\\Documents\\myrobot\\learning\model\\Logi_AL_20210827_145223.h5",
    #       host = "127.0.0.1:18300")
    serve(model_dir = "/home/xinyi/Workspace/myrobot/learning/model/Logi_AL_20210827_145223.h5",
          host = "localhost:50051")
          
          


