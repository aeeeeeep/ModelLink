import argparse
import random

import jsonlines
import torch
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig
from modelslim.pytorch.llm_ptq.anti_outlier import AntiOutlier, AntiOutlierConfig
from transformers import AutoTokenizer, TelechatForCausalLM, TelechatConfig


def quant(model, tokenizer):
    # random input
    data_num = 100
    calib_list = random.sample(questions, data_num)
    torch.save(calib_list, f"calib_list_{args.level}")

    # prepare calib data
    calib_data = []
    for text in calib_list:
        token_data = tokenizer(text, return_tensors="pt")
        calib_data.append([token_data["input_ids"].cpu(), None, token_data["attention_mask"].cpu()])
    
    # model to cpu
    model.cpu().float().eval()

    print("--------anti outlier suppression start--------")
    anti_config = AntiOutlierConfig(anti_method="m2", dev_type="cpu")
    anti_outlier = AntiOutlier(model, calib_data=calib_data, cfg=anti_config, model_type="Llama")
    anti_outlier.process()
    print("--------anti outlier suppression success--------")

    model.save_pretrained("telechat_anti_cpu")
    print("--------save anti outlier float weight success--------")

    print("-----------set quant config--------")
    quant_config = QuantConfig(w_bit=8, disable_names=[], dev_type='cpu', act_method=3, pr=1.0, mm_tensor=False, w_hessian=False)

    print("-----------init calibrator--------")
    calibrator = Calibrator(model, quant_config, calib_data=calib_data, disable_level=args.level)
    
    print("-----------calibrator run--------")
    calibrator.run(int_infer=True)
    
    print("-----------calibrator save--------")
    calibrator.save(f"anti_quant_weight_{args.level}")
    print("--------calibration end----------")
    return model


def get_args():
    parser = argparse.ArgumentParser(
        'Evaluation',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    group = parser.add_argument_group('EVAL Task Parameters')
    group.add_argument(
        '--level', type=str)
    group.add_argument(
        '--jsonl_path', type=str)
    group.add_argument(
        '--checkpoint_path', type=str)
    args = parser.parse_args()
    return args


args = get_args()

tokenizer = AutoTokenizer.from_pretrained(args.checkpoint_path)
config = TelechatConfig.from_pretrained(args.checkpoint_path)
model = TelechatForCausalLM.from_pretrained(args.checkpoint_path, config=config)

f = jsonlines.open(args.jsonl_path, "r")
questions = []
for data in f:
    questions.append(data["input"])
f.close()

model = quant(model, tokenizer) # anti outlier + ptq
print("quant model", model)