import os
import yaml

class BinConfig(object):
    def __init__(self, config_path=None):
        if config_path is None: 
            dir_path = os.path.dirname(os.path.realpath(__file__))
            config_path = os.path.realpath(os.path.join(dir_path, "../cfg/config.yaml"))
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
        self.data["hand"]["smc_length"] = (self.data["hand"]["smc"]["tool_changer_height"] + self.data["hand"]["smc"]["flange_height"] + self.data["hand"]["smc"]["gripper_height"] + self.data["hand"]["smc"]["finger_length"])/1000
        self.data["hand"]["schunk_length"] = (self.data["hand"]["schunk"]["tool_changer_height"] + self.data["hand"]["schunk"]["flange_height"] + self.data["hand"]["schunk"]["gripper_height"] + self.data["hand"]["schunk"]["finger_length"])/1000
        # depth = 10 * 255 / (self.data["pick"]["distance"]["max"] - self.config["pick"]["distance"]["min"])
        # self.data["graspability"]["hand_depth"] = int(depth)
        # self.write()

         
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
