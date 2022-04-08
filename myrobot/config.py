
import os
import yaml
import re
from collections import OrderedDict

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