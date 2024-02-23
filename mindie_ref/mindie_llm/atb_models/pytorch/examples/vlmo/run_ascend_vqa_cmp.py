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
from ais_bench.infer.interface import InferSession

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
    CLIP_DIR = "/home/jjfa/green_zone/vlmo/unilm/vlmo/vlmo/om/patchembed.om"
    VQA_ARROW_DIR = "/data1/models/vlmo/arrow/"
    BERT_VOCAB= "./vocab.txt"
    _config = copy.deepcopy(_config)
    database = VQAv2Dataset(image_size=_config["image_size"],data_dir=VQA_ARROW_DIR,transform_keys=_config["train_transform_keys"],split="test")
    database.tokenizer = BertTokenizer.from_pretrained(BERT_VOCAB)


    torch_npu.npu.set_device(DEVICE_ID )
    # dm = MTDataModule(_config, dist=True)
    
    # 注册om
    patchembed_model = InferSession(DEVICE_ID , CLIP_DIR)

    model = VLMo(_config).npu().half()
    # model.to(device=torch.device("npu:4"))
    model.eval()
    model.patchembed_model = patchembed_model

    same = 0
    notsame = 0
    notsametest = {}
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
        imgid = res["imgid"]
        question,other = res["text"]
        image = res["image"]
        print(str(i),"/",len(database)," qid:",qid," "," imgid:",imgid," ",question)
        runResult.write(" ".join((str(i),"/",str(len(database))," qid:",str(qid)," "," imgid:",str(imgid)," ",question,"\n")))
        # runResult.write(" ")
        # print("other",other)
        batch = {}
        batch["text_ids"] = torch.tensor( other["input_ids"]).reshape(1,database.max_text_len).npu()
        batch["text_masks"] = torch.tensor( other["attention_mask"]).reshape(1,database.max_text_len).npu()
        batch["text_labels"] = textlables.reshape(1,database.max_text_len).npu()
        imageTensor = []
        imageTensor.append( torch.tensor( image[0].reshape(1,3,480,480)).npu())
        batch["image"] = imageTensor
        batch["index"] = str(i)
    
    
        start_time_org = time.time()
        with torch.no_grad():
            
            infer_org = model.infer(batch, mask_text=False, mask_image=False)
            
            vqa_logits = model.vqa_classifier(infer_org["cls_feats"])
            # print("vqa_logits",vqa_logits)

            _, preds = vqa_logits.max(-1)
        end_time_org = time.time()
        res = label2ans[preds[0].item()]
        # print("org res: ", res)


        start_time_asc = time.time()
        with torch.no_grad():
            infer_asc = model.infer_ascend(batch, mask_text=False, mask_image=False)
            vqa_logits = model.vqa_classifier(infer_asc["cls_feats"])

            _, preds = vqa_logits.max(-1)

        end_time_asc = time.time()
        res2 = label2ans[preds[0].item()]


        if np.allclose(infer_org["cls_feats"].cpu(), infer_asc["cls_feats"].cpu(), rtol=0.02, atol=0.02): # 34.2  25.2 
            print("==> result equal.")
        else:
            print("==>!!!result not equal.")
            print(infer_org["cls_feats"])
            print(infer_asc["cls_feats"])
            exit(0)
       
        print("org cost:",float(end_time_org - start_time_org)*1000,"ms  ascend cost:",float(end_time_asc - start_time_asc)*1000,"ms")
        if res == res2:
            print("res same!!!")
            runResult.write(" ".join(("same", "org res:",res ,"ascend res:",res2,"\n")))
            same+=1
        else:
            print("res not same!!!")
            runResult.write(" ".join(("not same","org res:",res ," ascend res:",res2,"\n")))
            notsametest[batch["index"]] = (question,res,res2)
            notsame+=1
            runResultNotSame.write(" ".join(("qid:",str(qid),"imgid:",str(imgid),question,"not same","org res:",res ," ascend res:",res2,"\n")))
            # exit()

    print("same ",same, " notsame",notsame)
    runResult.write(" ".join(("same ",same," notsame",notsame,"\n")))
    for file in notsametest.keys():
        ques,res,res2= notsametest[file]
        runResult.write(" ".join(("index:",file,"quest:",ques,"res:",res,"res_ascend:",res2,"\n")))
        print("index:",file," quest:",ques," res:",res," res_ascend:",res2)

            



    # torch.save(infer,"infer_152054000_npu.pt")
    # infer_org = torch.load("infer_152054000.pt")
    

    # if np.allclose(infer_org["raw_cls_feats"].cpu(), infer["raw_cls_feats"].cpu(), rtol=0.08, atol=0.08):
    #     print("==>  equal.")
    # else:
    #     print("==>!!!not equal .")
    #     print(infer_org["raw_cls_feats"])
    #     print(infer["raw_cls_feats"])
    

    # trainer = pl.Trainer(
    #     gpus=_config["num_gpus"],
    #     num_nodes=_config["num_nodes"],
    #     precision=_config["precision"],
    #     accelerator="npu",
    #     strategy=distributed_strategy,
    #     benchmark=True,
    #     deterministic=True,
    #     max_epochs=_config["max_epoch"] if max_steps is None else 1000,
    #     max_steps=max_steps,
    #     callbacks=callbacks,
    #     logger=logger,
    #     # prepare_data_per_node=False,
    #     replace_sampler_ddp=False,
    #     accumulate_grad_batches=grad_steps,
    #     log_every_n_steps=10,
    #     flush_logs_every_n_steps=10,
    #     resume_from_checkpoint=resume_ckpt,
    #     weights_summary="top",
    #     fast_dev_run=_config["fast_dev_run"],
    #     val_check_interval=_config["val_check_interval"],
    #     plugins=plugin_list,
    # )

    # if _config["loss_names"]["textmlm"] > 0:
    #     for param in model.parameters():
    #         param.requires_grad = False

    #     for name, param in model.named_parameters():
    #         for key in ["text_embeddings", "token_type_embeddings", "mlp_text", "norm2_text", "mlm_score", "relative_position_bias_table", "transformer.norm"]:
    #             if key in name:
    #                 param.requires_grad = True

    #     for name, param in model.named_parameters():
    #         rank_zero_info("{}\t{}".format(name, param.requires_grad))

    # if not _config["test_only"]:
    #     trainer.fit(model, datamodule=dm)
    # else:
    #     trainer.test(model, datamodule=dm)
