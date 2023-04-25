import os
import yaml
import numpy as np

class BinConfig(object):
    def __init__(self, config_path=None, pre=True):
        self.root_dir = os.path.realpath(os.path.join(os.path.realpath(__file__), "../.."))
        self.depth_dir = os.path.join(self.root_dir, "data/depth")
        if config_path is None: 
            config_path = os.path.join(self.root_dir, "cfg/config.yaml")
        hand_path = os.path.join(self.root_dir, "cfg/hand.yaml")
        self.hand_path = hand_path

        self.motionfile_path = os.path.join(self.root_dir, "data/motion/motion.dat")
        self.data_path = config_path
        self.data = {}
        self.load()
        if pre:
            self.pre_define()
    
    def load(self):
        try:
            with open(self.data_path) as f:
                self.data = yaml.load(f, Loader=yaml.FullLoader)
        except FileNotFoundError:
            print('Wrong file or file path')

    def pre_define(self):
        """unit: mm -> m"""
        try:
            with open(self.hand_path) as f:
                hand = yaml.load(f, Loader=yaml.FullLoader)
                lhand = hand[self.data["hand"]["left"]["type"]]
                rhand = hand[self.data["hand"]["right"]["type"]]
        except FileNotFoundError:
            print("Wrong file or file path")

        left_len, right_len = 0, 0
        for k, i in lhand.items():
            if "height" in k:
                left_len += i
        for k, i in rhand.items():
            if "height" in k:
                right_len += i
        
        self.data["hand"]["left"]["height"]=left_len/1000
        self.data["hand"]["right"]["height"]=right_len/1000
        self.data["hand"]["left"]["real2pixel"] = self.data["real2pixel"]
        self.data["hand"]["right"]["real2pixel"] = self.data["real2pixel"]
        self.data["hand"]["left"].update(lhand)
        self.data["hand"]["right"].update(rhand)


    def get_all_values(self):
        for v in self.data.values():
            if isinstance(v, dict):
                yield from self.get_all_values(v)
            else:
                yield v

    def set(self, keys, values):
        if isinstance(keys, str):
            self.data[keys] = values

    def write(self):
        try:
            with open(self.data_path, 'w') as f:
                yaml.dump(self.data, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print('Wrong file or file path')

    def update(self, key, value, write=False):
        if isinstance(value, np.floating): 
            value = float(value)
        self.data[key] = value
        if write:
            self.write()

    def keys(self):
        return self.data.keys()
