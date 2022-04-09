from cProfile import label
from cmath import pi
from operator import index
import os
import sys
import math
import random
from datetime import datetime as dt
from turtle import color

from numpy import save

from bprobot.binpicking import *
from bprobot.prediction import predict_client as pdclt
pdc = pdclt.PredictorClient()

"""Draft!!!!"""

def count_illogical():
    s_logical = 0
    s_ill = 0
    f_logical = 0
    f_ill = 0

    s, f = 0,0

    root_dir = "/home/xinyi/Workspace/bprobot_bak/dataset/picking_400"

    for name in os.listdir(root_dir):

        print(name)

        _, graspx, graspsy, angle, action, label_ = name.split('_')
        label = int(label_.split('.')[0])
        action = int(action)

        # fp = open("/home/xinyi/n.txt", 'a')
        # print(f"{action},{label}",file=fp)
        # fp.close()

        img_path = os.path.join(root_dir, name)
        grasps = [[None, graspx, graspsy, None, float(angle)]]
        grasps2bytes=np.ndarray.tobytes(np.array(grasps))
        predict_result= pdc.predict(imgpath=img_path, grasps=grasps2bytes)
        best_action = predict_result.action

        if label == 1:
            s+=1 
            # success
            if best_action > action:
                s_ill+=1
                print("y")
            elif best_action <= action:
                s_logical+1
        elif label == 0:
            f+=1
            if best_action < action:
                f_ill +=1 
            else: 
                f_logical +=1
    
    print(f"success: {s}, logical: {s_logical}, illogical: {s_ill}" )
    print(f"fail: {f}, logical: {f_logical}, illogical: {f_ill}" )

def test():
    h_params = (finger_height, finger_width, open_width, hand_template_size)
    root_dir = "/home/xinyi/Workspace/bprobot_bak/exp/proposed_method/long_for_paper"
    img_name = "20210830145030_274_281_6_.png"
    img_path = os.path.join(root_dir, img_name)
    _, graspx, graspsy, angle, label_ = img_name.split('_')
    angle = 0
    # _, graspx, graspsy, angle, action, label_ = img_name.split('_')

    grasps = [[None, graspx, graspsy, None, float(angle)]]
    grasps2bytes=np.ndarray.tobytes(np.array(grasps))
    predict_result= pdc.predict(imgpath=img_path, grasps=grasps2bytes)
    best_action = predict_result.action
    print(best_action)
    
    
    img = cv2.imread(img_path)
    img = adjust_grayscale(img, max_b=220)
    img = draw_grasps(grasps, 0, img, h_params, (255,0,0))
    cv2.imshow("window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_path = os.path.join(root_dir, "vis_"+img_name)
    cv2.imwrite(save_path, img)

def vis():

    # colors = [(255,0,0),(255,99,0),(255,213,0),(71,255,184),(0,212,255)]
    colors = [(255,0,0),(255,111,0),(255,177,0)]
    # grasps = [[[None, 577, 381, None, float(90)]],
    #           [[None, 220, 55, None, float(67.5)]],
    #           [[None, 106, 247, None, float(45)]],
    #           [[None, 664, 570, None, float(67.5)]]]
    grasps = [[[None, 582, 445, None, float(68)]],
            [[None, 224, 543, None, float(67.5)]],
            [[None, 517, 580, None, float(90)]]]
            # [[None, 664, 570, None, float(67.5)]]]
    h_params = (finger_height, finger_width, open_width, hand_template_size)

    # print(img_name)

    # _, graspx, graspsy, angle, action, label_ = img_name.split('_')
    # label = int(label_.split('.')[0])
    # action = int(action)

    # fp = open("/home/xinyi/n.txt", 'a')
    # print(f"{action},{label}",file=fp)
    # fp.close()

    # grasps = [[None, graspx, graspsy, None, float(angle)]]
    # grasps2bytes=np.ndarray.tobytes(np.array(grasps))
    # predict_result= pdc.predict(imgpath=img_path, grasps=grasps2bytes)
    # best_action = predict_result.action

    img = cv2.imread(img_path)
    
    for i in range(3):
        grasp = grasps[i]
        img = draw_grasps(grasps[i], 0, img, h_params, colors[i])

        grasps2bytes=np.ndarray.tobytes(np.array(grasp))
        predict_result= pdc.predict(imgpath=img_path, grasps=grasps2bytes)
        best_action = predict_result.action

        print(best_action)
    cv2.imshow("window", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    save_path = os.path.join(root_dir, "vis_"+img_name)
    print(save_path)
        # save_path = f"/home/xinyi/val_{graspx}_{graspsy}_{action}_{label}_{best_action}.png"
    # print(save_path)
    # cv2.imwrite(save_path, draw_grasp_img)


def main():
    
    main_proc_print("Start! ")
    # ========================== define path =============================
    root_dir = "/home/xinyi/Workspace/bprobot_bak/exp/proposed_method/long_for_paper"
    root_dir = "/home/xinyi/Workspace/dataset/val_long"

    from bprobot.prediction import predict_client as pdclt
    
    pdc = pdclt.PredictorClient()
    for name in os.listdir(root_dir):

        print(name)

        _, graspx, graspsy, _,_ = name.split('_')


        img_path = os.path.join(root_dir, name)
        grasps = [[None, graspx, graspsy, None, float(angle)]]
        grasps2bytes=np.ndarray.tobytes(np.array(grasps))
        predict_result= pdc.predict(imgpath=img_path, grasps=grasps2bytes)
        best_action = predict_result.action

    
    # rotation_step = 22.5
    # depth_step = 50
    # hand_depth = 50
    # # ===========================================================================++++++++++++++++++++++++++++++++++++++++++++++

    # finger_w=13
    # finger_h=40
    # open_w = 96/2-20
    # gripper_size = 500
    # colors = [(255,0,0),(255,111,0),(255,177,0),(192,255,61),(78,255,177)]
    # # grasps = [[[None, 577, 381, None, float(90)]],
    # #           [[None, 220, 55, None, float(67.5)]],
    # #           [[None, 106, 247, None, float(45)]],
    # #           [[None, 664, 570, None, float(67.5)]]]
    # grasps = [[[None, 384, 510, None, float(100)*3.14/180]],
    #         [[None, 315, 175, None, float(135)*3.14/180]],
    #         [[None, 646, 594, None, float(135)*3.14/180]],
    #         [[None, 136, 548, None, float(45)*3.14/180]]]
    #         # [[None, 65, 53, None, float(135)*3.14/180]]]
    #         # [[None, 195, 418, None, float(0)]]]
    #         # [[None, 664, 570, None, float(67.5)]]]
    # h_params = (finger_height, finger_width, open_width, hand_template_size)

    # img = cv2.imread(img_path)
    
    # for i in range(len(grasps)):
    #     grasp = grasps[i]
    #     img = draw_grasps(grasps[i], 0, img, h_params, colors[i])

    #     grasps2bytes=np.ndarray.tobytes(np.array(grasp))
    #     predict_result= pdc.predict(imgpath=img_path, grasps=grasps2bytes)
    #     best_action = predict_result.action
    #     print(best_action)
    # cv2.imshow("window", img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # save_path = os.path.join(root_dir, "vis_"+img_name)
    # print(save_path)
    
    # ===========================================================================++++++++++++++++++++++++++++++++++++++++++++++


    # finger_w=6.5
    # finger_h=20
    # open_w = 40
    # gripper_size = 500
    # main_proc_print("Rotation step: {}".format(rotation_step))
    # main_proc_print("Depth step: {}".format(depth_step))

    # margins = (0,0,1080,869)
    # g_params = (rotation_step, depth_step, hand_depth)
    # h_params = (finger_h, finger_w, open_w, gripper_size)

    # grasps, input_img, full_image = detect_grasp_point(n_grasp=5, img_path=img_path, margins=margins, g_params=g_params, h_params=h_params)
    # # grasps, input_img, full_image = detect_target_oriented_grasp(10, ROOT_DIR, margins, g_params, h_params)

    # # temporal printing
    # if grasps:
    #     result_print(f"Top grasp: pixel location=({grasps[0][1]},{grasps[0][2]}), angle={grasps[0][4]*180/math.pi}, width={grasps[0][-1]}")

    # if grasps:
    #     for i in range(len(grasps)):
    #         result_print(f"Top grasp={grasps[i][0]}: pixel location=({grasps[i][1]},{grasps[i][2]}), angle={grasps[i][4]*180/math.pi}, width={grasps[i][-1]}")
    
    # grasps2bytes=np.ndarray.tobytes(np.array(grasps))
    # predict_result= pdc.predict(imgpath=img_path, grasps=grasps2bytes)
    # best_action = predict_result.action
    # best_graspno = predict_result.graspno
    # best_grasp = grasps[predict_result.graspno]
    
    # print(best_action)
    
    # cv2.imshow("windows", full_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()



    # ===========================================================================++++++++++++++++++++++++++++++++++++++++++++++
  
    # main_proc_print("Save the results! ")


if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
