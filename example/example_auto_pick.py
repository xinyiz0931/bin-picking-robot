import os
import argparse
import random

from bpbot.binpicking import *
from bpbot.config import BinConfig

def main():
    
    main_proc_print("Start! ")
    # ========================== define path =============================
    # get root dir
    #root_dir = os.path.abspath("./")
    root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
    
    depth_dir = os.path.join(root_dir, "data/depth/")

    # pc_path = os.path.join(root_dir, "data/pointcloud/out.ply")
    img_path = os.path.join(root_dir, "data/depth/depth.png")
    # img_path = os.path.join(root_dir, "data/test/depth9.png")
    crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
    config_path = os.path.join(root_dir, "cfg/config.yaml")
    calib_path = os.path.join(root_dir, "data/calibration/calibmat.txt")
    mf_path = os.path.join(root_dir, "data/motion/motion.dat")
    draw_path = os.path.join(root_dir, "data/depth/result.png")
  
    cfg = BinConfig(config_path)

    # ======================= get parameters ============================
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", help="mode of autopicking",default="capture", choices=['capture','file'])
    parser.add_argument("--file", "-f", help="path to image", default=str(img_path))
    args = parser.parse_args()
    # ======================= get config info ============================

    bincfg = BinConfig(config_path)
    cfg = bincfg.config
    flag = False
    
    # ======================== get depth img =============================

    if args.mode == 'capture':
        point_array = get_point_cloud(depth_dir, cfg['pick']['distance'],
                                      cfg['width'],cfg['height'])
        if point_array is None: return
        # pcd = o3d.io.read_point_cloud("./data/test/out.ply")
        # point_array = pcd.points

    else:
        img_path = args.file
        point_array = None
    # =======================  compute grasp =============================

    grasps, img_input = detect_grasp_point(n_grasp=10, 
                                   img_path=img_path, 
                                   margins=cfg['pick']['margin'],
                                   g_params=cfg['graspability'],
                                   h_params=cfg["hand"])
    cv2.imwrite(crop_path, img_input)
    # cv2.imwrite(draw_path, img_grasp)

    # =======================  picking policy ===========================
    if grasps is None:
        best_action = -1
        best_graspno = 0
        # rx,ry,rz,ra = np.zeros(4)
        return
    else:

        if cfg['exp_mode'] == 0:
            # 0 -> graspaiblity
            best_graspno = 0
            best_grasp = grasps[best_graspno]
            best_action = 0 

        elif cfg['exp_mode'] == 1: 
            # 1 -> proposed circuclar picking
            grasp_pixels = np.array(grasps)[:, 1:3]
            best_action, best_graspno = predict_action_grasp(grasp_pixels, crop_path)
            best_grasp = grasps[best_graspno]

        elif cfg['exp_mode'] == 2:
            # 2 -> random circular picking
            best_grasp = grasps[0]
            best_action = random.sample(list(range(6)),1)[0]
    
        (rx,ry,rz,ra) = transform_image_to_robot((best_grasp[1],best_grasp[2],best_grasp[4]),
                        img_path, calib_path, point_array, cfg["pick"]["margin"])
    # draw grasp 

    img_grasp = draw_grasps(grasps, crop_path,  cfg["hand"], top_idx=best_graspno, top_color=(255,0,0), top_only=True)
    cv2.imwrite(draw_path, img_grasp)

    # =======================  generate motion ===========================
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

    
