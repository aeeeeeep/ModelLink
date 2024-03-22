import argparse
import json
import os.path
import time
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
    parser.add_argument("--npu-id", type=int, default=0,
                        help="Specify the npu to load the model.")
    parser.add_argument("--cfg-path", default="eval_configs/minigpt4_eval.yaml",
                        help="Path to configuration file.")
    parser.add_argument("--image-path", required=True,
                        help="Image path for inference.")
    parser.add_argument("--output-path", required=True,
                        help="Output path of inference.")
    parser.add_argument("--options", nargs="+",
                        help="override some settings in the used config, the key-value pair "
                             "in xxx=yyy format will be merged into config file (deprecate), "
                             "change to --cfg-options instead.",
                        )
    return parser.parse_args()


def set_torch_env(device_ids):
    torch_npu.npu.set_device(int(device_ids))
    torch.npu.set_compile_mode(jit_compile=False)


def traverse_img_dir(img_dir, res_file_dir):
    # 判断目标目录是否存在
    if not os.path.exists(img_dir):
        print("目标目录不存在！")
        return
    if not os.path.exists(res_file_dir):
        os.mkdir(res_file_dir)

    input_text = "Describe this image in detail."

    for root, dirs, files in os.walk(img_dir):
        if not files:
            continue
        cur_time = time.strftime("%m%d_%H%M", time.localtime())
        res_file_name = f"res_{cur_time}_{os.path.basename(root)}.json"
        res_file_path = os.path.join(res_file_dir, res_file_name)
        res_dict = {"image_file": [], "model_answer": []}
        for file in files:
            image_path = os.path.join(root, file)
            print("文件路径：", image_path)
            chat_state = CONV_VISION_Vicuna0.copy()
            img_list = []
            llm_message = chat.upload_img(image_path, chat_state, img_list)
            print(f"{llm_message=}")
            chat.encode_img(img_list)
            print(f"===== image_list: {img_list}")
            print(f"===== chat_state: {chat_state.messages}")
            chat.ask(input_text, chat_state)
            llm_message = chat.answer(conv=chat_state,
                                      img_list=img_list,
                                      num_beams=1,
                                      temperature=0.1,
                                      max_new_tokens=300,
                                      max_length=2000)[0]
            print(f"MiniGPT4 Answer: {llm_message}")
            res_dict["image_file"].append(image_path)
            res_dict["model_answer"].append(llm_message)
            n_saved = len(res_dict["image_file"])
            print(f"已生成 {n_saved} 条记录 from {root}")
        with open(res_file_path, "w") as f:
            json.dump(res_dict, f)
    print("-----ALL DONE-----")


if __name__ == '__main__':
    # Model Initialization
    print('Initializing Chat')
    args = parse_args()
    set_torch_env(args.npu_id)
    cfg = Config(args)

    model_config = cfg.model_cfg
    model_config.device_8bit = args.npu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'npu:{args.npu_id}')

    vis_processor_cfg = cfg.datasets_cfg.cc_sbu_align.vis_processor.train
    vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)

    stop_words_ids = [[835], [2277, 29937]]
    stop_words_ids = [torch.tensor(ids).to(device=f'npu:{args.npu_id}') for ids in stop_words_ids]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])

    chat = Chat(model, vis_processor, device=f'npu:{args.npu_id}', stopping_criteria=stopping_criteria)
    print('Initialization Finished')

    # Model Inference
    traverse_img_dir(img_dir=args.image_path, res_file_dir=args.output_path)
