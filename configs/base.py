import os
import json
import yaml


class Config(object):

    def __init__(self, project_name):
        self.project_name = project_name
        self.base_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "projects")
        self.config_dict = {
            "System": {
                "Project": None,
                "GPU": True,
                "GPU_ID": 0,
                "Allow_Ext": ["jpg", "jpeg", "png", "bmp"],
                "Path": "",
                "Val": 0.03
            },
            "Model": {
                "ImageWidth": -1,
                "ImageHeight": 64,
                "ImageChannel": 1,
                "CharSet": [],
                "Word": False
            },
            "Train": {
                "BATCH_SIZE": 32,
                "TEST_BATCH_SIZE": 32,
                'CNN': {
                    "NAME": "ddddocr",
                },
                'DROPOUT': 0.3,
                'OPTIMIZER': 'SGD',
                "TEST_STEP": 1000,
                "TARGET": {
                    "Accuracy": 0.97,
                    "Epoch": 200,
                    "Cost": 0.005
                },
                "LR": 0.01
            }
        }


    def make_config(self, config_dict=None, single: bool = False):
        if not config_dict:
            config_dict = self.config_dict
            if single:
                config_dict['Model']['Word'] = True

            config_dict["System"]["Project"] = self.project_name
        config_path = os.path.join(self.base_path, self.project_name, "config.yaml")
        with open(config_path, 'w', encoding="utf-8") as f:
            yaml.dump(config_dict, f, allow_unicode=True, default_flow_style=None, indent=4)

    def load_config(self):
        config_path = os.path.join(self.base_path, self.project_name, "config.yaml")
        with open(config_path, 'r', encoding="utf-8") as f:
            config_dict = yaml.load(f, Loader=yaml.FullLoader)
        return config_dict
