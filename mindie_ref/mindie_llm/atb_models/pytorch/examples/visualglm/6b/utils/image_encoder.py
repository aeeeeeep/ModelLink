import os

from ais_bench.infer.interface import InferSession


def singleton(cls):
    instances = {}
    def wrapper(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]
    return wrapper


@singleton
class IMAGE_ENCODER_OM:
    def __init__(self, model_path, device_id):
        blip_path = os.path.join(model_path, 'blip2.om')
        self.image_encoder_om = InferSession(device_id, blip_path)  # read local om_file
        