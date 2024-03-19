import argparse
import os
import copy
import time

import json
from transformers import BertTokenizer
import torch
import torch_npu
from vlmo.config import ex
from vlmo.modules.vlmo_module_cut import VLMo
from vlmo.datasets import VQAv2Dataset


def label_2_ans(path):
    ans2label_file = os.path.join(path, "answer2label.txt")
    ans2label = {}
    label2ans = []
    with open(ans2label_file, mode="r", encoding="utf-8") as reader:
        for i, line in enumerate(reader):
            data = json.loads(line)
            ans = data["answer"]
            label = data["label"]
            label = int(label)
            ans2label[ans] = i
            label2ans.append(ans)
    return label2ans


@ex.automain
def main(_config):
    _config = copy.deepcopy(_config)
    device_Id = _config["device"]
    VQA_ARROW_DIR = "./vlmo/arrow/"
    BERT_VOCAB = "./vlmo/vocab.txt"
    LOAD_PATH = "./vlmo/"
    PT_NAME = "cut_VQA_weights.pt"
    local_rank = 0
    database = VQAv2Dataset(
        image_size=_config["image_size"],
        data_dir=VQA_ARROW_DIR,
        transform_keys=_config["val_transform_keys"],
        split="test",
    )
    database.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)

    if len(device_Id) == 1:
        torch_npu.npu.set_device(device_Id[0])
        _config["load_path"] = os.path.join(LOAD_PATH, PT_NAME)
        model = VLMo(_config).npu().half()
        model.eval()
    else:
        torch.distributed.init_process_group("hccl")
        local_rank = torch.distributed.get_rank()
        world_size = torch.distributed.get_world_size()
        part_model_path = os.path.join(LOAD_PATH, 'part_model', str(local_rank), PT_NAME)
        _config["load_path"] = part_model_path
        _config["test_only"] = True
        print(_config["load_path"])
        torch_npu.npu.set_device(device_Id[local_rank])
        model = VLMo(_config).npu().half()
        model.eval()

    textlables = torch.full((database.max_text_len,), -101, dtype=torch.int)
    label2ans = label_2_ans(VQA_ARROW_DIR)
    for res in database:
        qid = res["qid"]
        question, other = res["text"]
        image = res["image"]
        print(" qid:", qid, " ", question)
        
        batch = {}
        batch["text_ids"] = (
            torch.tensor(other["input_ids"]).reshape(1, database.max_text_len).npu()
        )
        batch["text_masks"] = (
            torch.tensor(other["attention_mask"])
            .reshape(1, database.max_text_len)
            .npu()
        )
        batch["text_labels"] = textlables.reshape(1, database.max_text_len).npu()
        imageTensor = []
        imageTensor.append(torch.tensor(image[0].reshape(1, 3, 480, 480)).npu())
        batch["image"] = imageTensor
        with torch.no_grad():
            start_time_org = time.time()
            infer = model.infer(batch, mask_text=False, mask_image=False)
            end_time_org = time.time()
            vqa_logits = model.vqa_classifier(infer["cls_feats"])
            _, preds = vqa_logits.max(-1)
        res = label2ans[preds[0].item()]
        print("cost:", float(end_time_org - start_time_org) * 1000, "ms")
        print("res:", res)