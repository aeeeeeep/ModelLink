import os
import copy
import time

import json
from transformers import BertTokenizer
import torch
import torch_npu
from vlmo.config import ex
from vlmo.modules.vlmo_module import VLMo
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
            assert label == i
            ans2label[ans] = i
            label2ans.append(ans)
    return label2ans

@ex.automain
def main(_config):
    
    DEVICE_ID = 4
    VQA_ARROW_DIR = "/data1/models/vlmo/arrow/"
    BERT_VOCAB = "./vocab.txt"
    _config = copy.deepcopy(_config)
    database = VQAv2Dataset(image_size=_config["image_size"],data_dir=VQA_ARROW_DIR,transform_keys=_config["train_transform_keys"],split="test")
    database.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)


    torch_npu.npu.set_device(DEVICE_ID )
    # dm = MTDataModule(_config, dist=True)
    
    # 注册om

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
    label2ans = label_2_ans(VQA_ARROW_DIR)
    for i in range(len(database)):
        # not same :  1315 5120
        # if i < 5121:
        #     continue
        res = database[i]
        qid = res["qid"]
        question,other = res["text"]
        image = res["image"]
        print(str(i),"/",len(database)," qid:",qid," ",question)
        runResult.write(" ".join((str(i),"/",str(len(database))," qid:",str(qid)," ",question,"\n")))
        batch = {}
        batch["text_ids"] = torch.tensor( other["input_ids"]).reshape(1,database.max_text_len).npu()
        batch["text_masks"] = torch.tensor( other["attention_mask"]).reshape(1,database.max_text_len).npu()
        batch["text_labels"] = textlables.reshape(1,database.max_text_len).npu()
        imageTensor = []
        imageTensor.append( torch.tensor( image[0].reshape(1,3,480,480)).npu())
        batch["image"] = imageTensor
        batch["index"] = str(i)
    
        with torch.no_grad():
            start_time_org = time.time()
            infer = model.infer_ascend(batch, mask_text=False, mask_image=False)
            end_time_org = time.time()
            vqa_logits = model.vqa_classifier(infer["cls_feats"])
            # print("vqa_logits",vqa_logits)
            
            _, preds = vqa_logits.max(-1)
        res = label2ans[preds[0].item()]
        # print("org res: ", res)

        print("cost:",float(end_time_org - start_time_org)*1000,"ms")

        print("res:",res)
        runResult.write(" ".join(("res:",res,"\n")))
         