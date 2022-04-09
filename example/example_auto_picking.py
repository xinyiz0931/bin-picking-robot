import os
import sys
import math
import configparser
import random
from datetime import datetime as dt

from matplotlib.pyplot import margins
from bpbot.binpicking import *
from bpbot.config import BinConfig

# def main():
#     root_dir = os.path.abspath("./")
#     config_path = os.path.join(root_dir, "cfg/config.yaml")
#     cfg = BinConfig(config_path)
#     print(cfg.test_no)

#     cfg.set("test_no", 81)

#     # cfg.update("test_no", "this")
#     print(cfg.test_no)

def main():
    
    main_proc_print("Start! ")
    # ========================== define path =============================
    root_dir = os.path.abspath("./")
    calib_dir = os.path.join(root_dir, "data/calibration/")
    depth_dir = os.path.join(root_dir, "data/depth/")

    # pc_path = os.path.join(root_dir, "data/pointcloud/out.ply")
    img_path = os.path.join(root_dir, "data/depth/depth.png")
    img_path = os.path.join(root_dir, "data/depth/depth_raw.png")
    crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
    config_path = os.path.join(root_dir, "cfg/config.yaml")
    calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
    mf_path = os.path.join(root_dir, "data/motion/motion.dat")
    draw_path = os.path.join(root_dir, "data/depth/result.png")

    # ======================= get config info ============================
    cfg = BinConfig(config_path)

    # ======================== get depth img =============================
    point_array = get_point_cloud(depth_dir, cfg.max_distance, cfg.min_distance, cfg.width, cfg.height)
    #pcd = o3d.io.read_point_cloud("./data/test/out.ply")
    #point_array = pcd.points

    # =======================  compute grasp =============================
    grasps, img_input, img_grasp = detect_grasp_point(n_grasp=10, 
                                                       img_path=img_path, 
                                                       margins=cfg.margins, 
                                                       g_params=cfg.g_params, 
                                                       h_params=cfg.h_params)
    cv2.imwrite(crop_path, img_input)
    cv2.imwrite(draw_path, img_grasp)

    # =======================  picking policy ===========================
    if grasps is None:
        best_action = -1
        best_graspno = 0
        rx,ry,rz,ra = np.zeros(4)
    else:
        # four policies
        if cfg.exp_mode == 0:
            # 0 -> graspaiblity
            best_grasp = grasps[0]
            best_action = 0 

        elif cfg.exp_mode == 1: 
            # 1 -> proposed circuclar picking
            # from bprobot.prediction import predict_client as pdclt
            # pdc = pdclt.PredictorClient()
            # grasps2bytes=np.ndarray.tobytes(np.array(grasps))
            # predict_result= pdc.predict(imgpath=crop_path, grasps=grasps2bytes)
            # best_action = predict_result.action
            # best_graspno = predict_result.graspno
            best_action, best_graspno = predict_action_grasp(grasps, crop_path)
            best_grasp = grasps[best_graspno]
            # decrease the time cost
            # if best_action == 6: best_action = random.sample([3,4],1)[0]
            # if best_action == 0: best_action = 1
        elif cfg.exp_mode == 2:
            # 2 -> random circular picking
            best_grasp = grasps[0]
            best_action = random.sample(list(range(6)),1)[0]

        # rx,ry,rz,ra = transform_coordinates(best_grasp, point_array, img_path, calib_path, cfg.width, cfg.margins)
        (rx,ry,rz,ra) = transform_image_to_robot((best_grasp[1],best_grasp[2],best_grasp[4]),
                                                img_path, calib_path, point_array, cfg.margins)

    # # =======================  generate motion ===========================
    success_flag = generate_motion(mf_path, [rx,ry,rz,ra], best_action) 
  
    # ======================= Record the data ===================s=========
    main_proc_print("Save the results! ")

    # if success_flag:
    #     tdatetime = dt.now()
    #     tstr = tdatetime.strftime('%Y%m%d%H%M%S')
    #     input_name = "{}_{}_{}_{}_.png".format(tstr,best_grasp[1],best_grasp[2],best_action)
    #     cv2.imwrite('./exp/{}'.format(input_name), input_img)
    #     cv2.imwrite('./exp/draw/{}'.format(input_name), cimg)
    #     cv2.imwrite('./exp/full/{}'.format(input_name), full_image)

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
