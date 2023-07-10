'''''''''
A Python scripts to set the workspace size 
Author: xinyi
Date: 20210721
---
Define workspace for bin picking task and update config file!
Step 1 - Drag to select 2 or 4 rectangles and press [ENTER]
Step 2 - Click to select 2 or 4 points and press [ENTER]
[r] - refresh and reselect
[q] - quit
'''
import argparse
import cv2
from bpbot.config import BinConfig
from bpbot.device import PhxClient
from bpbot.utils import *

class Workspace(object):
    def __init__(self, cfg_path=None):
        self.cfg = BinConfig(config_path=cfg_path, pre=False)
        self.cfgdata = self.cfg.data
        self.vis = None 
        self.clone = None

        self.boxes = []
        self.points = []
        self.mat = np.loadtxt(self.cfgdata['calibmat_path'])

    def capture(self):
        pxc = PhxClient(host='127.0.0.1:18300')
        pxc.triggerframe()
        pc = pxc.getpcd()
        self.pc = pc
        
        self.grayscale = pxc.getgrayscaleimg()
        self.clone = cv2.cvtColor(self.grayscale, cv2.COLOR_GRAY2BGR)
        self.vis = self.clone.copy()
        
        pc = pc/1000
        H = np.loadtxt(self.cfgdata['calibmat_path'])
        pc_ = np.c_[pc, np.ones(pc.shape[0])]
        pr = np.dot(H, pc_.T).T
        self.arr = np.reshape(pr[:,2], (self.cfgdata['height'], self.cfgdata['width']))

    def recog_ar_marker(self, ids=[7]): 
        self.capture()
        image = cv2.cvtColor(self.grayscale, cv2.COLOR_GRAY2RGB)
        recog_ids = detect_ar_marker(image.copy(), show=True)
        cv2.waitKey(2)
        for id in ids:
            if id in recog_ids.keys():
                x,y = recog_ids[id]
                p = self.pc[y*image.shape[1]+x]/1000

                p_robot = np.dot(self.mat, [*p, 1])  # unit: m
                p_robot = p_robot[:3]
                print(f"=> Detected #{id}!")
                print(f"   Camera ({p[0]:.3f},{p[1]:.3f},{p[2]:.3f})")
                print(f"   Robot  ({p_robot[0]:.3f},{p_robot[1]:.3f},{p_robot[2]:.3f})")
            else: 
                print("Marker #{id} not detected")
        

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
            cv2.namedWindow('Click')
            cv2.setMouseCallback('Click', self.on_click, 0)
            cv2.imshow('Click', self.vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('r'):
                self.refresh_click()
            if key == ord('q'):
                cv2.destroyAllWindows()
                break
            if key == ord('a'):
                self.auto_define()
                break

            if len(self.points) <= 4 and key == 13: # enter
                cv2.destroyAllWindows()
                break

    def auto_define(self):
        self.points = []
        top_id = 28
        bottom_ids = [21,22]
        real_d = 20+20+29/3

        image = cv2.cvtColor(self.grayscale, cv2.COLOR_GRAY2RGB)
        ids = detect_ar_marker(image.copy(), show=True)
        cv2.waitKey(2)
        if top_id in ids:
            x,y = ids[top_id]
            pick_max = self.arr[y,x]
            print(f"Detected top: {x},{y}, => {pick_max:.3f}")
        if bottom_ids[0] in ids and bottom_ids[1] in ids:
            x0, y0 = ids[bottom_ids[0]]
            x1, y1 = ids[bottom_ids[1]]
            pixel_d = calc_2points_distance((x0,y0),(x1,y1))
            pick_min = (self.arr[y0,x0]+self.arr[y1,x1])/2
            print(f"Detected bottom: {x},{y}, => {pick_min:.3f}")
        
            self.cfgdata['main']['height']['min'] = float(pick_min)-0.002
            self.cfgdata['main']['height']['max'] = float(pick_max)
            self.cfgdata['real2pixel'] = float(pixel_d/real_d)
            self.cfg.write()

            print("Successfully defined **height** of picking workspace! ")

    def select_boxes(self):
        self.boxes = []
        while(True):
            cv2.namedWindow('Drag')
            cv2.setMouseCallback('Drag', self.on_drag, 0)
            cv2.imshow('Drag', self.vis)
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
            self.cfgdata['main']['area']['left'] = self.boxes[0][0]
            self.cfgdata['main']['area']['top'] = self.boxes[0][1]
            self.cfgdata['main']['area']['right'] = self.boxes[1][0]
            self.cfgdata['main']['area']['bottom'] = self.boxes[1][1]
            self.cfgdata['buffer']['area']['left'] = self.boxes[2][0]
            self.cfgdata['buffer']['area']['top'] = self.boxes[2][1]
            self.cfgdata['buffer']['area']['right'] = self.boxes[3][0]
            self.cfgdata['buffer']['area']['bottom'] = self.boxes[3][1]
            self.cfg.write()
            print("Successfully defined **area** of picking workspace! ")
            print("Successfully defined **area** of dropping workspace! ")

        elif len(self.boxes) == 2:
            self.cfgdata['main']['area']['left'] = self.boxes[0][0]
            self.cfgdata['main']['area']['top'] = self.boxes[0][1]
            self.cfgdata['main']['area']['right'] = self.boxes[1][0]
            self.cfgdata['main']['area']['bottom'] = self.boxes[1][1]
            self.cfg.write()
            print("Successfully defined **area** of picking workspace! ")

        else:
            print("Did not calibrate area of workspace! ")

        if len(self.points) == 4:
            h = [self.arr[self.points[0][1], self.points[0][0]], self.arr[self.points[1][1], self.points[1][0]]]
            [pick_min, pick_max] = h if h[0] < h[1] else [h[1], h[0]]

            h = [self.arr[self.points[2][1], self.points[2][0]], self.arr[self.points[3][1], self.points[3][0]]]
            [drop_min, drop_max] = h if h[0] < h[1] else [h[1], h[0]]
            self.cfgdata['main']['height']['min'] = float(pick_min)
            self.cfgdata['main']['height']['max'] = float(pick_max)
            self.cfgdata['buffer']['height']['min'] = float(drop_min)
            self.cfgdata['buffer']['height']['max'] = float(drop_max)
            self.cfg.write()
            print("Successfully defined **height** of picking workspace! ")
            print("Successfully defined **height** of dropping workspace! ")

        elif len(self.points) == 2:
            h = [self.arr[self.points[0][1], self.points[0][0]], self.arr[self.points[1][1], self.points[1][0]]]
            [pick_min, pick_max] = h if h[0] < h[1] else [h[1], h[0]]
            self.cfgdata['main']['height']['min'] = float(pick_min)
            self.cfgdata['main']['height']['max'] = float(pick_max)
            self.cfg.write()
            print("Successfully defined **height** of picking workspace! ")



def main():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('mode', type=str, help='marker or none')
    parser.add_argument('--file_path','-f', type=str, help='if you want to save a new config file, enter the file path')
    args = parser.parse_args()
    
    ws = Workspace(args.file_path)

    if args.mode == 'marker':
        ws.recog_ar_marker()
    else:
        print("Define workspace for bin picking task and update config file! ")
        print("Step 1 - Drag to select 2 or 4 rectangles and press [ENTER]")
        print("Step 2 - Click to select 2 or 4 points and press [ENTER]")
        print("[r] - refresh and reselect")
        print("[q] - quit")
        print("[a] - automatically define the height")
        
        ws.capture()
        ws.select()
        ws.define()

if __name__ == '__main__':
    main()
