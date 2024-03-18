import os
import copy
import time

import json
import numpy as np
from transformers import BertTokenizer
import torch
import torch_npu
from vlmo.config import ex
from vlmo.modules.vlmo_module import VLMo
from vlmo.datasets import VQAv2Dataset
from vlmo.datasets import CocoCaptionKarpathyDataset


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

    DEVICE_ID = 4
    COCO_ARROW_DIR = "./vlmo/cocoarrow/"
    BERT_VOCAB = "./vocab.txt"
    _config = copy.deepcopy(_config)
    database = CocoCaptionKarpathyDataset(
        image_size=_config["image_size"],
        data_dir=COCO_ARROW_DIR,
        transform_keys=_config["train_transform_keys"],
        split="test",
    )
    database.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)

    torch_npu.npu.set_device(DEVICE_ID)
    model = VLMo(_config).npu().half()
    model.eval()

    textlables = torch.full((database.max_text_len,), -101, dtype=torch.int)
    for res in database:

        text, other = res["text"]
        imgid = res["iid"]
        image = res["image"]

        print("iid:", imgid, " ", text)
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

        imageTensor.append(torch.tensor(image[0].reshape(1, 3, 384, 384)).npu().half())
        batch["image"] = imageTensor
        with torch.no_grad():
            start_time_org = time.time()
            infer_org = model.infer_text_ft(batch, mask_text=False)
            infer_asc = model.infer_text_ft_ascend(batch, mask_text=False)
            end_time_org = time.time()
            print("cost:", float(end_time_org - start_time_org) * 1000, "ms")
            if np.allclose(
                infer_org["cls_feats"].cpu(),
                infer_asc["cls_feats"].cpu(),
                rtol=0.02,
                atol=0.02,
            ):
                print("==> text result equal.")
            else:
                print("==>!!!text result not equal.")
                print(infer_org["cls_feats"])
                print(infer_asc["cls_feats"])

        with torch.no_grad():
            start_time_org = time.time()
            infer_org = model.infer_image_ft(batch, mask_image=False)
            infer_asc = model.infer_image_ft_ascend(batch, mask_image=False)
            end_time_org = time.time()
            print("cost:", float(end_time_org - start_time_org) * 1000, "ms")
            if np.allclose(
                infer_org["cls_feats"].cpu(),
                infer_asc["cls_feats"].cpu(),
                rtol=0.02,
                atol=0.02,
            ):
                print("==>image  result equal.")
            else:
                print("==>!!!image result not equal.")
                print(infer_org["cls_feats"])
                print(infer_asc["cls_feats"])
