import os
import yaml

class BinConfig(object):
    def __init__(self, config_path):
        self.config_path = config_path
        self.config = {}
        self.load()
        # self.pre_define()
    
    def load(self):
        try:
            with open(self.config_path) as f:
                self.config = yaml.load(f, Loader=yaml.FullLoader)
            # self.load_yaml(config_path)
        except FileNotFoundError:
            print('Wrong file or file path')
        # import ruamel.yaml
        # doc = ruamel.yaml.load(config_path, Loader=ruamel.yaml.RoundTripLoader)

        # print(doc['test'])
        # doc['test'] = 'byebye world'

        # with open(config_path, 'w+', encoding='utf8') as outfile:
        #     ruamel.yaml.dump(doc, outfile,Dumper=ruamel.yaml.RoundTripDumper)

    def pre_define(self):
        """unit: cm"""
        depth = 10 * 255 / (self.config["pick"]["distance"]["max"] - self.config["pick"]["distance"]["min"])
        self.config["graspability"]["hand_depth"] = int(depth)
        self.write()
         
    # def __getattr__(self, key):
    #     try:
    #         return self.config[key]
    #     except KeyError:
    #         raise AttributeError

    def set(self, key, value):
        self.config[key] = value

    def write(self):
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print('Wrong file or file path')

    def update(self, key, value):
        self.config[key] = value
        try:
            with open(self.config_path, 'w') as f:
                yaml.dump(self.config, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print('Wrong file or file path')

    def keys(self):
        return self.config.keys()