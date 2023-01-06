import numpy as np
<<<<<<< Updated upstream
# import matplotlib.pyplot as plt
# from scipy.misc import electrocardiogram
# from scipy.signal import find_peaks
# x = electrocardiogram()[2000:4000]
# print(x.shape)
# peaks, _ = find_peaks(x, height=0)
# plt.plot(x)
# print(peaks.shape)
# plt.plot(peaks, x[peaks], "x")
# plt.plot(np.zeros_like(x), "--", color="gray")
# plt.show()
# import matplotlib.pyplot as plt
# from scipy.interpolate import UnivariateSpline
# rng = np.random.default_rng()
# x = np.linspace(-3, 3, 50)
# y = np.exp(-x**2) + 0.1 * rng.standard_normal(50)
# plt.plot(x, y, 'ro', ms=5)

# spl = UnivariateSpline(x, y)
# xs = np.linspace(-3, 3, 1000)
# plt.plot(xs, spl(xs), 'g', lw=3)

# spl.set_smoothing_factor(0.5)
# plt.plot(xs, spl(xs), 'b', lw=3)
# plt.show()
import matplotlib.pyplot as plt
import numpy as np
mu, sigma = 0, 500
x = np.arange(1, 100, 0.1)  # x axis
z = np.random.normal(mu, sigma, len(x))  # noise
y = x ** 2 + z # data
from scipy.signal import savgol_filter
w = savgol_filter(y, 51, 2)
plt.plot(x, y, linewidth=2, linestyle="-", c='r')  # it include some noise
plt.plot(x, w, 'b')  # high frequency noise removed

plt.show()
=======
from bpbot.config import BinConfig
from bpbot.binpicking import *
import pickle
import cv2


def detect(img_path, exp_mode):
    h_params = {
        "finger_length": 15,
        "finger_width":  8, 
        "open_width":    30
    }
    # h_params = {
    #     "finger_length": 15,
    #     "finger_width":  8, 
    #     "open_width":    30
    # }
    g_params = {
        "rotation_step": 22.5, 
        "depth_step":    25,
        "hand_depth":    25
    }
    t_params = {
        "compressed_size": 200,
        "len_thld": 10,
        "dist_thld": 5,
        "sliding_size": 50,
        "sliding_stride": 50
    }
    print(f"[*] Method : {exp_mode}")
    if exp_mode.lower() == "fge":
        grasps = detect_grasp(n_grasp=1, img_path=img_path, 
                        g_params=g_params,
                        h_params=h_params)
        print("test:", grasps)
        if grasps is not None: 
            # return grasps[0][:2]
            return grasps
        return

    elif exp_mode.lower() == "emap": 
        out = detect_nontangle_grasp(n_grasp=1, img_path=img_path, 
                                    g_params=g_params, 
                                    h_params=h_params,
                                    t_params=t_params)
        if out is not None: 
            return out
            # return grasps[0][:2]
        return
    
    elif exp_mode.lower() == "pn":
        from bpbot.module_picksep import PickSepClient
        psc = PickSepClient()
        ret = psc.infer_picknet(img_path)
        return ret[1][:2]
    else: 
        print("[!] No valid mode! ")
        return
cfg = BinConfig()
cfgdata = cfg.data
root_dir = "C:\\Users\\xinyi\\Material\\RA-L2022Tangle\\result\\vis_for_ral\\pn_sr"
print(pickle.load(open(os.path.join(root_dir, "picknet_score.pickle"), 'rb')))

img_path = os.path.join(root_dir, "depth_cropped_pickbin.png")

out = detect(img_path, 'emap')
g, emap = out
img = cv2.imread(img_path)

# vis = cv2.applyColorMap(emap, cv2.COLORMAP_JET)
vis = cv2.applyColorMap(emap, cv2.COLORMAP_VIRIDIS)
overlay = cv2.addWeighted(img, 0.85, vis, 0.15, 0)
img_grasp = draw_grasp(g, overlay, cfgdata["hand"]["left"], top_color=(0,255,0), top_only=True)
cv2.imshow("", img_grasp)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite(os.path.join(root_dir, "emap2.png"), img_grasp)


# out = detect(img_path, 'fge')

# img_grasp = draw_grasp(out, img_path, cfgdata["hand"]["left"], top_color=(0,255,0), top_only=True)

# cv2.imshow("", img_grasp)
# cv2.waitKey()
# cv2.destroyAllWindows()
# cv2.imwrite(os.path.join(root_dir, "fge.png"), img_grasp)




>>>>>>> Stashed changes
