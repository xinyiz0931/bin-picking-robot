import os
import matplotlib.pyplot as plt
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
    print(f"[*] Start scrip at {root_dir}! ")
    
    depth_dir = os.path.join(root_dir, "data/depth/")
    img_path = os.path.join(depth_dir, "depth.png") 
    save_path = os.path.join(depth_dir, "capture.png")
    cfg = BinConfig()
    cfg = cfg.data
    H = np.loadtxt(cfgdata["calibmat_path"])
    
    # ---------------------- get depth img -------------------------
    pc = capture_pc()
    # img, img_blur = px2depth(pc, cfg)
    # crop_pb = crop_roi(img_path, cfgdata["pick"]["area"])
    # crop_db = crop_roi(img_path, cfgdata["drop"]["area"])
    
    # plt.imshow(crop_pb, cmap='gray'), plt.show()
    # plt.imshow(crop_db, cmap='gray'), plt.show()

    pc /= 1000
    # pc = np.ones((3186816, 3))
    # print(pc.shape)
    pc_ = np.c_[pc, np.ones(pc.shape[0])]
    pr = np.dot(H, pc_.T).T
    print(pr.shape)
    gray_array = pr[:,2]
    max_, min_ = 0.044, -0.060 
    max_, min_ = 0.044, 0.003 
    print(max_, min_)
    gray_vis = gray_array.reshape((cfgdata["height"], cfgdata["width"]))
    vis = cv2.normalize(gray_vis, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8) 
    cv2.imwrite(save_path, vis)
    plt.imshow(vis), plt.show()
    # gray_array[gray_array > max_] = max_
    # gray_array[gray_array < min_] = min_
    # # gray_array = - gray_array
    # img = ((gray_array - min_) *
    #        (1/(max_ - min_) * 255)).astype('uint8')
    # img = img.reshape((cfgdata["height"], cfgdata["width"]))
    # plt.imshow(img), plt.show()
    
    # get_point_cloud(depth_dir, cfgdata[args.zone]["height"],
    #                               cfgdata["width"],cfgdata["height"])
    
    # ---------------------- visualize -------------------------
    # crop = crop_roi(img_path, cfgdata[args.zone]["area"])
    # cv2.imshow("cropped image", crop)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

if __name__ == "__main__":

    import timeit
    start = timeit.default_timer()

    main()
    
    end = timeit.default_timer()
    print("[*] Time: {:.2f}s".format(end - start))

    
