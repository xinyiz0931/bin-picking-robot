import os
from datetime import datetime as dt

from bpbot.binpicking import *
from bpbot.config import BinConfig

def main():
    main_proc_print("Start! ")
    # ========================== define path =============================
    root_dir = os.path.abspath("./")
    depth_dir = os.path.join(root_dir, "data/depth/")
    img_path = os.path.join(root_dir, "data/depth/depth.png")
    gs_path = os.path.join(root_dir, "data/depth/texture.png")
    crop_path = os.path.join(root_dir, "data/depth/depth_cropped.png")
    config_path = os.path.join(root_dir, "cfg/config.yaml")
    # ======================= get config info ============================
    bincfg = BinConfig(config_path)
    cfg = bincfg.config
    # ======================== get depth img =============================
    point_array = get_point_cloud(depth_dir, cfg['mid']['distance'],
                                  cfg['width'],cfg['height'])
    
    # ======================== crop depth image =============================
    crop = crop_roi(img_path, cfg['mid']['margin'])
    cv2.imshow("cropped image", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite(crop_path, crop)

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
