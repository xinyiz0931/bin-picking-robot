img_path = "C:\\Users\\xinyi\\Material\\RAL2022Tangle\\tangle_exp_for_ral\\PNOnly\\exp_se\\test1\\20221201115946\\depth_drop.png"
img_path = "C:\\Users\\xinyi\\Material\\RAL2022Tangle\\tangle_exp_for_ral\\PNOnly\\exp_se\\test1\\20221201115527\\depth_drop.png"
from bpbot.binpicking import *
from bpbot.config import BinConfig
cfg = BinConfig()
cfgdata = cfg.data
ret_dropbin = pick_or_sep(img_path=img_path, hand_config=cfgdata["hand"], bin='drop')
_, g_pull,v_pull = ret_dropbin
img_grasp = draw_pull_grasps(img_path, g_pull, v_pull)
cv2.imshow("", img_grasp)
cv2.waitKey()
cv2.destroyAllWindows()
cv2.imwrite("C:\\Users\\xinyi\\Desktop\\a.png", img_grasp)