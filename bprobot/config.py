
import os
import yaml

class BinConfig(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
        try:
            with open(config_path) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            # self.load_yaml(config_path)
        except FileNotFoundError:
            print("Wrong file or file path")

    def __getattr__(self, key):
        try:
            if key== "margins":
                return (self.config["top_margin"],
                        self.config["left_margin"],
                        self.config["bottom_margin"],
                        self.config["right_margin"])
            elif key== "g_params":
                return (self.config["rotation_step"],
                        self.config["depth_step"],
                        self.config["hand_depth"])
            elif key== "h_params":
                return (self.config["finger_width"],
                        self.config["finger_height"],
                        self.config["gripper_width"],
                        self.config["hand_template_size"])
            elif key== "t_params":
                return (self.config["length_thre"],
                        self.config["distance_thre"],
                        self.config["sliding_size"],
                        self.config["sliding_stride"],
                        self.config["compressed_size"])
            else:
                return self.config[key]
        except KeyError:
            raise AttributeError

    def set(self, key, value):
        self.config[key] = value

    def write(self):
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print("Wrong file or file path")

    def update(self, key, value):
        self.config[key] = value
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print("Wrong file or file path")

    def keys(self):
        return self.config.keys()
    