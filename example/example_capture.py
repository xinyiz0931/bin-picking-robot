import os
from datetime import datetime as dt

from bpbot.binpicking import *
from bpbot.config import BinConfig

def main():
    main_proc_print("Start! ")
    # ========================== define path =============================
    ROOT_DIR = os.path.abspath("./")
    depth_dir = os.path.join(ROOT_DIR, "data/depth/")
    img_path = os.path.join(ROOT_DIR, "data/depth/depth.png")
    crop_path = os.path.join(ROOT_DIR, "data/depth/depth_cropped.png")

    # ======================= get config info ============================
    cfg = BinConfig()
    # ======================== get depth img =============================
    get_point_cloud(depth_dir, cfg.max_distance, cfg.min_distance, cfg.width, cfg.height)
    
    # ======================== crop depth image =============================
    crop = crop_roi(img_path, cfg.margins)
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

    
