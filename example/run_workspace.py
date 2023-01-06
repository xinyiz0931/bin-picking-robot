"""
A Python scripts to set the workspace size 
Author: xinyi
Date: 20210721
---
Define workspace for bin picking task and update config file!
Step 1 - Drag to select 2 or 4 rectangles and press [ENTER]
Step 2 - Click to select 2 or 4 points and press [ENTER]
[r] - refresh and reselect
[q] - quit
"""
import cv2
from bpbot.config import BinConfig
from bpbot.device import PhxClient
from bpbot.utils import *

class Workspace(object):
    def __init__(self):
        self.cfg = BinConfig(pre=False)
        self.cfgdata = self.cfg.data
        self.vis = None 
        self.clone = None

        self.boxes = []
        self.points = []
        self.mat = np.loadtxt(self.cfgdata["calibmat_path"])

    def capture(self):
        pxc = PhxClient(host='127.0.0.1:18300')
        pxc.triggerframe()
        pc = pxc.getpcd()
        
        self.grayscale = pxc.getgrayscaleimg()
        self.clone = cv2.cvtColor(self.grayscale, cv2.COLOR_GRAY2BGR)
        self.vis = self.clone.copy()
        
        pc = pc/1000
        H = np.loadtxt(self.cfgdata["calibmat_path"])
        pc_ = np.c_[pc, np.ones(pc.shape[0])]
        pr = np.dot(H, pc_.T).T
        self.arr = np.reshape(pr[:,2], (self.cfgdata["height"], self.cfgdata["width"]))
    
    def refresh_drag(self): 
        self.vis = self.clone.copy()
        self.boxes.clear()
    def refresh_click(self):
        self.vis = self.clone.copy()
        self.points.clear()

    def on_drag(self, event, x, y, flags, params):
        # global img
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Start Mouse Position: {x},{y}")
            sbox = [x, y]
            self.boxes.append(sbox)

        elif event == cv2.EVENT_LBUTTONUP:
            print(f"End Mouse Position: {x},{y}")
            ebox = [x, y]
            self.boxes.append(ebox)
            cv2.rectangle(self.vis, (self.boxes[-2][0], self.boxes[-2][1]), (self.boxes[-1][0], self.boxes[-1][1]), (0,255,0), 3)
    
    def on_click(self, event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            print(f"Mouse Position: {x},{y}, => {self.arr[y,x]:.3f}")
            cv2.circle(self.vis,(x,y),5,(0,255,0),-1)
            self.points.append([x,y])
        
    def select_points(self):
        self.points = []
        while(True):
            cv2.namedWindow("Click")
            cv2.setMouseCallback("Click", self.on_click, 0)
            cv2.imshow("Click", self.vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.refresh_click()
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            if len(self.points) <= 4 and key == 13: # enter
                cv2.destroyAllWindows()
                break

    def select_boxes(self):
        self.boxes = []
        while(True):
            cv2.namedWindow("Drag")
            cv2.setMouseCallback("Drag", self.on_drag, 0)
            cv2.imshow("Drag", self.vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.refresh_drag()
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            if len(self.boxes)/2 <= 2 and key == 13: # enter
                cv2.destroyAllWindows()
                break

    def select(self):
        self.select_boxes()
        self.select_points()

    def define(self):
        if len(self.boxes) == 4:
            self.cfgdata["pick"]["area"]["left"] = self.boxes[0][0]
            self.cfgdata["pick"]["area"]["top"] = self.boxes[0][1]
            self.cfgdata["pick"]["area"]["right"] = self.boxes[1][0]
            self.cfgdata["pick"]["area"]["bottom"] = self.boxes[1][1]
            self.cfgdata["drop"]["area"]["left"] = self.boxes[2][0]
            self.cfgdata["drop"]["area"]["top"] = self.boxes[2][1]
            self.cfgdata["drop"]["area"]["right"] = self.boxes[3][0]
            self.cfgdata["drop"]["area"]["bottom"] = self.boxes[3][1]
            print("Successfully defined **area** of picking workspace! ")
            print("Successfully defined **area** of dropping workspace! ")

        elif len(self.boxes) == 2:
            self.cfgdata["pick"]["area"]["left"] = self.boxes[0][0]
            self.cfgdata["pick"]["area"]["top"] = self.boxes[0][1]
            self.cfgdata["pick"]["area"]["right"] = self.boxes[1][0]
            self.cfgdata["pick"]["area"]["bottom"] = self.boxes[1][1]
            print("Successfully defined **area** of picking workspace! ")

        else:
            print("Did not calibrate area of workspace! ")

        if len(self.points) == 4:
            h = [self.arr[self.points[0][1], self.points[0][0]], self.arr[self.points[1][1], self.points[1][0]]]
            [pick_min, pick_max] = h if h[0] < h[1] else [h[1], h[0]]

            h = [self.arr[self.points[2][1], self.points[2][0]], self.arr[self.points[3][1], self.points[3][0]]]
            [drop_min, drop_max] = h if h[0] < h[1] else [h[1], h[0]]
            self.cfgdata["pick"]["height"]["min"] = float(pick_min)
            self.cfgdata["pick"]["height"]["max"] = float(pick_max)
            self.cfgdata["drop"]["height"]["min"] = float(drop_min)
            self.cfgdata["drop"]["height"]["max"] = float(drop_max)
            print("Successfully defined **height** of picking workspace! ")
            print("Successfully defined **height** of dropping workspace! ")

        elif len(self.points) == 2:
            h = [self.arr[self.points[0][1], self.points[0][0]], self.arr[self.points[1][1], self.points[1][0]]]
            [pick_min, pick_max] = h if h[0] < h[1] else [h[1], h[0]]
            self.cfgdata["pick"]["height"]["min"] = float(pick_min)
            self.cfgdata["pick"]["height"]["max"] = float(pick_max)
            print("Successfully defined **height** of picking workspace! ")

        else:
            print("Did not define height of workspace! ")
            return 

        self.cfg.write()

def main():

    print("Define workspace for bin picking task and update config file! ")
    print("Step 1 - Drag to select 2 or 4 rectangles and press [ENTER]")
    print("Step 2 - Click to select 2 or 4 points and press [ENTER]")
    print("[r] - refresh and reselect")
    print("[q] - quit")
    
    ws = Workspace()
    ws.capture()
    ws.select()
    ws.define()

if __name__ == '__main__':
    main()
