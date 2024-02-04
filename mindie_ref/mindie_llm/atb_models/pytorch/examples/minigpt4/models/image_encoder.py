import os

from ais_bench.infer.interface import InferSession


def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


# @singleton
class IMAGE_ENCODER_OM:
    def __init__(self, model_path, device):
        eva_vit_path = os.path.join(model_path, 'eva_vit_g.om')
        self.image_encoder_om = InferSession(device, eva_vit_path)  # read local om_file
