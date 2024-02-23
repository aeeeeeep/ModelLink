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
            assert label == i
            ans2label[ans] = i
            label2ans.append(ans)
    return label2ans

@ex.automain
def main(_config):
    
    DEVICE_ID = 4
    CLIP_DIR = "/home/jjfa/green_zone/vlmo/unilm/vlmo/om/patchembedcoco.om"
    COCO_ARROW_DIR = "/data1/models/vlmo/cocoarrow/"
    BERT_VOCAB= "./vocab.txt"
    _config = copy.deepcopy(_config)
    # database = VQAv2Dataset(image_size=_config["image_size"],data_dir=VQA_ARROW_DIR,transform_keys=_config["train_transform_keys"],split="test")
    # database.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)
    # 1 3 384 384
    database = CocoCaptionKarpathyDataset(image_size=_config["image_size"],data_dir=COCO_ARROW_DIR,transform_keys=_config["train_transform_keys"],split="test")
    database.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)

    torch_npu.npu.set_device(DEVICE_ID)
    # dm = MTDataModule(_config, dist=True)

    model = VLMo(_config).npu().half()
    # model.to(device=torch.device("npu:4"))
    model.eval()

    textlables = torch.full(
            (database.max_text_len,),
            -101,
            dtype=torch.int
        )
    runResult = open("multtest.log", "w")
    runResultNotSame = open("multtest_notsame.log", "w")
    # label2ans = label_2_ans(VQA_ARROW_DIR)
    for i in range(len(database)):
        # not same :  1315 5120
        # if i < 5121:
        #     continue
        # print(database[i])
        res = database[i]

        text,other = res["text"]

        imgid = res["iid"]
        
        image = res["image"]


        print(str(i),"/",len(database)," iid:",imgid," ",text)
        # runResult.write(" ".join((str(i),"/",str(len(database))," qid:",str(qid)," "," imgid:",str(imgid)," ",question,"\n")))
        # runResult.write(" ")
        # print("other",other)
        batch = {}
        batch["text_ids"] = torch.tensor( other["input_ids"]).reshape(1,database.max_text_len).npu()
        batch["text_masks"] = torch.tensor( other["attention_mask"]).reshape(1,database.max_text_len).npu()
        batch["text_labels"] = textlables.reshape(1,database.max_text_len).npu()
        imageTensor = []
        
        imageTensor.append( torch.tensor( image[0].reshape(1,3,384,384)).npu().half())
        batch["image"] = imageTensor
        batch["index"] = str(i)
        
        # model.current_tasks = ["irtr"]
        # print("current_tasks",model.current_tasks)
        # ret = model(batch)
        # print("ret",ret)
    
        with torch.no_grad():
            start_time_org = time.time()
            infer_org = model.infer_text_ft(batch, mask_text=False)
            
            # print("infer_org",infer_org["cls_feats"])

            infer_asc = model.infer_text_ft_ascend(batch, mask_text=False)
            
            # print("infer_asc",infer_asc["cls_feats"])
            end_time_org = time.time()
            print("cost:",float(end_time_org - start_time_org)*1000,"ms")
            if np.allclose(infer_org["cls_feats"].cpu(), infer_asc["cls_feats"].cpu(), rtol=0.02, atol=0.02): # 34.2  25.2 
                print("==> text result equal.")
            else:
                print("==>!!!text result not equal.")
                print(infer_org["cls_feats"])
                print(infer_asc["cls_feats"])
                exit(0)
        
        with torch.no_grad():
            start_time_org = time.time()
            infer_org = model.infer_image_ft(batch, mask_image=False)
            # print("infer_org shape",infer_org["cls_feats"].shape)
            # print("infer_org",infer_org["cls_feats"])

            infer_asc = model.infer_image_ft_ascend(batch, mask_image=False)
            # print("infer_asc shape",infer_asc["cls_feats"].shape)
            # print("infer_asc",infer_asc["cls_feats"])
            end_time_org = time.time()
            print("cost:",float(end_time_org - start_time_org)*1000,"ms")
            if np.allclose(infer_org["cls_feats"].cpu(), infer_asc["cls_feats"].cpu(), rtol=0.02, atol=0.02): # 34.2  25.2 
                print("==>image  result equal.")
            else:
                print("==>!!!image result not equal.")
                print(infer_org["cls_feats"])
                print(infer_asc["cls_feats"])
                exit(0)