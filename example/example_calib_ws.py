"""
A Python scripts to set the workspace size 
Author: xinyi
Date: 20210721
Usage: `python example/example_calib_ws.py dist_margin -z pick_mid`
---
Draw a rectangle for inside of the box. 
If you are happy with the size, hit `enter` or `q`, the config file will be updated. 
If you want re-draw the rectangle, hit `r` to refresh. 
"""
import bpbot.driver.phoxi.phoxi_client as pclt
import os
import cv2
import argparse

from bpbot.config import BinConfig
from bpbot.utils import *

def shape_selection(event, x, y, flags, param):
    global ref_point, crop

    if event == cv2.EVENT_LBUTTONDOWN:
        ref_point = [(x, y)]

    elif event == cv2.EVENT_LBUTTONUP:
        ref_point.append((x, y))

        cv2.rectangle(image, ref_point[0], ref_point[1], (0, 255, 0), 3)

parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["dist", "margin", "all"])

parser.add_argument("--zone", "-z", help="capture which zone",
                    choices=["pick", "mid", "pick_mid"], default="pick")

args = parser.parse_args()

root_dir = os.path.abspath("./")
config_path = os.path.join(root_dir, "cfg/config.yaml")
cfg = BinConfig(config_path)
min_dist_id, max_dist_id = 1, 10

ws_mode = "dist_margin" if args.mode == "all" else args.mode

pxc = pclt.PhxClient(host="127.0.0.1:18300")
pxc.triggerframe()
gray = pxc.getgrayscaleimg()
image = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
clone = image.copy()

if "dist" in ws_mode and args.zone != "pick_mid":
    main_proc_print(f"Detect distances using marker for {args.zone}")
    pcd = pxc.getpcd()
    pcd_r = rotate_point_cloud(pcd)
    id_locs = detect_ar_marker(image.copy())

    if id_locs == {}:
        warning_print("[*] No markers! Failed to define distances! ")
    else:
        main_proc_print(f"[*] Detected markers: {id_locs}")

    if min_dist_id in id_locs.keys():
        w = image.shape[1]
        x, y, = id_locs[min_dist_id]
        # offset = y * w + x
        min_distance = int(pcd_r[y*w+x][-1])
        notice_print(f"min_distance:  {min_distance}")

    if max_dist_id in id_locs.keys():
        w = image.shape[1]
        x, y, = id_locs[max_dist_id]
        # offset = y * w + x
        max_distance = int(pcd_r[y*w+x][-1])
        notice_print(f"max_distance:  {max_distance}")

    cfg.config[args.zone]["distance"]["max"] = max_distance
    cfg.config[args.zone]["distance"]["min"] = min_distance - 15
    cfg.write()
    main_proc_print("Successfully defined the max/min distance! ")

if "margin" in ws_mode:
    main_proc_print(f"Generating workspace for {args.zone}")
    if "pick" in args.zone:
        cv2.namedWindow("Define pick zone", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Define pick zone", shape_selection)
        cv2.resizeWindow("Define pick zone", 1920, 1080)

        while True:
            cv2.putText(image, "Drag the mouse for pick zone and click enter",
                        (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("Define pick zone", image)
            key = cv2.waitKey(1) & 0xFF

            # press r to reset the window
            if key == ord("r"):
                image = clone.copy()

            elif key == ord("q"):
                flag = 0
                break
            elif key == 13:  # enter
                flag = 1
                break
        cv2.destroyAllWindows()

        if flag:
            notice_print(f"left_margin:   {ref_point[0][0]}")
            notice_print(f"top_margin:    {ref_point[0][1]}")
            notice_print(f"right_margin:  {ref_point[1][0]}")
            notice_print(f"bottom_margin: {ref_point[1][1]}")

            cfg.config["pick"]["margin"]["left"] = ref_point[0][0]
            cfg.config["pick"]["margin"]["top"] = ref_point[0][1]
            cfg.config["pick"]["margin"]["right"] = ref_point[1][0]
            cfg.config["pick"]["margin"]["bottom"] = ref_point[1][1]
            cfg.write()

            main_proc_print("Successfully defined pick workspace size! ")
        else:
            warning_print("Failed to define pick workspace size! ")

    if "mid" in args.zone:
        cv2.namedWindow("Define mid zone", cv2.WINDOW_NORMAL)
        cv2.setMouseCallback("Define mid zone", shape_selection)
        cv2.resizeWindow("Define mid zone", 1920, 1080)

        while True:
            cv2.putText(image, "Drag the mouse for mid zone and click enter",
                        (80, 160), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            cv2.imshow("Define mid zone", image)
            key = cv2.waitKey(1) & 0xFF
            # press r to reset the window
            if key == ord("r"):
                image = clone.copy()

            elif key == ord("q"):
                flag = False
                break
            elif key == 13:  # enter
                flag = True
                break

        cv2.destroyAllWindows()

        if flag:
            notice_print(f"left_margin:   {ref_point[0][0]}")
            notice_print(f"top_margin:    {ref_point[0][1]}")
            notice_print(f"right_margin:  {ref_point[1][0]}")
            notice_print(f"bottom_margin: {ref_point[1][1]}")

            cfg.config["mid"]["margin"]["left"] = ref_point[0][0]
            cfg.config["mid"]["margin"]["top"] = ref_point[0][1]
            cfg.config["mid"]["margin"]["right"] = ref_point[1][0]
            cfg.config["mid"]["margin"]["bottom"] = ref_point[1][1]
            cfg.write()

            main_proc_print("Successfully defined mid workspace size! ")
        else:
            warning_print("Failed to define mid workspace size! ")
