import os
import argparse
from bpbot.binpicking import * 
from bpbot.config import BinConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zone", "-z", help="capture which zone? ", 
                        choices=["pick", "drop"], default="pick")

    args = parser.parse_args()

    # ----------------- define path and config ---------------------
    root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../../"))
    main_proc_print(f"Start scrip at {root_dir}! ")
    
    depth_dir = os.path.join(root_dir, "data/depth/")
    img_path = os.path.join(depth_dir, "depth.png") 
    
    bincfg = BinConfig()
    cfg = bincfg.data
    
    # ---------------------- get depth img -------------------------
    get_point_cloud(depth_dir, cfg[args.zone]["distance"],
                                  cfg["width"],cfg["height"])
    
    # ---------------------- visualize -------------------------
    crop = crop_roi(img_path, cfg[args.zone]["margin"])
    cv2.imshow("cropped image", crop)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    main_proc_print("Time: {:.2f}s".format(end - start))

    
