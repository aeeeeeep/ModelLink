import argparse
import random

import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from transformers import StoppingCriteriaList

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, CONV_VISION_Vicuna0, CONV_VISION_LLama2, StoppingCriteriaSub

from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--cfg-path", required=True, help="Path to configuration file.")
    parser.add_argument("--image-path", required=True, help="Image path for inference.")
    parser.add_argument("--npu-id", type=int, default=0, help="Specify the npu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    return parser.parse_args()


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def set_torch_env(device_ids):
    torch_npu.npu.set_device(int(device_ids))
    torch.npu.set_compile_mode(jit_compile=False)


if __name__ == '__main__':
    # Model Initialization
    conv_dict = {'pretrain_vicuna0': CONV_VISION_Vicuna0,
                 'pretrain_llama2': CONV_VISION_LLama2}

    print('Initializing Chat')
    args = parse_args()
    cfg = Config(args)

    set_torch_env(args.npu_id)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.npu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to('npu:{}'.format(args.npu_id))

    CONV_VISION = conv_dict[model_config.model_type]

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device='npu:{}'.format(args.npu_id)) for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device='npu:4', stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    # Model Inference
    image_path = args.image_path
    input_text = "Describe this image in detail."

    chat_state = CONV_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(image_path, chat_state, img_list)
    chat.encode_img(img_list)

    chat.ask(input_text, chat_state)
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=1,
                              temperature=1.0,
                              max_new_tokens=300,
                              max_length=2000)[0]
    print(f"MiniGPT4 Answer: {llm_message}")
