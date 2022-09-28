import os
import yaml

class BinConfig(object):
    def __init__(self, config_path=None):

        dir_path = os.path.dirname(os.path.realpath(__file__))
        if config_path is None: 
            config_path = os.path.realpath(os.path.join(dir_path, "../cfg/config.yaml"))
        hand_path = os.path.realpath(os.path.join(dir_path, "../cfg/hand.yaml"))
        self.hand_path = hand_path

        self.data_path = config_path
        self.data = {}
        self.load()
        self.pre_define()

        # self.pre_define()
    
    def load(self):
        try:
            with open(self.data_path) as f:
                self.data = yaml.load(f, Loader=yaml.FullLoader)
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
        self.data["hand"]["left"].update(lhand)
        self.data["hand"]["right"].update(rhand)
        # depth = 10 * 255 / (self.data["pick"]["distance"]["max"] - self.config["pick"]["distance"]["min"])
        # self.data["graspability"]["hand_depth"] = int(depth)

         
    # def __getattr__(self, key):
    #     try:
    #         return self.data[key]
    #     except KeyError:
    #         raise AttributeError

    def set(self, key, value):
        self.data[key] = value

    def write(self):
        try:
            with open(self.data_path, 'w') as f:
                yaml.dump(self.data, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print('Wrong file or file path')

    def update(self, key, value):
        self.data[key] = value
        try:
            with open(self.data_path, 'w') as f:
                yaml.dump(self.data, stream=f,
                        default_flow_style=False, sort_keys=False)
        except FileNotFoundError:
            print('Wrong file or file path')

    def keys(self):
        return self.data.keys()
