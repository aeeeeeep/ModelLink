# 导入相关依赖
import torch
import torch.utils.data
from transformers import AutoTokenizer, AutoModelForCausalLM
from modelslim.pytorch.llm_ptq.llm_ptq_tools import Calibrator, QuantConfig # 导入量化配置接口


def inference(_model, _tokenizer, max_new_tokens=32):
    test_prompt = "def Fibonacci_sequence(n):"
    test_input = _tokenizer(test_prompt, return_tensors="pt")
    print("model is inferring...")
    _model.eval()
    generate_ids = _model.generate(test_input.input_ids.cpu(), attention_mask=test_input.attention_mask.cpu(), max_new_tokens=max_new_tokens)
    res = _tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    for idx, item in enumerate(res):
        print(item)


#获取校准数据函数定义
def get_calib_dataset(_tokenizer, _calib_list):
    calib_dataset = []
    for calib_data in _calib_list:
        inputs = _tokenizer([calib_data], return_tensors='pt').to('cpu')
        print(inputs)
        calib_dataset.append([inputs.data['input_ids'], None, inputs.data['attention_mask']])
    return calib_dataset

# for local path
tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path='/home/data/starcoder')
model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path='/home/data/starcoder', torch_dtype=torch.float32).cpu()
print(f"loading success!")
print("start quant...")

# 准备校准数据，请根据实际情况修改
calib_list = []
with open('humaneval_python.txt', 'r') as file:
    for line in file:
        calib_list.append(line.strip())
#校准数据获取
dataset_calib = get_calib_dataset(tokenizer, calib_list) 

# 量化配置
# 配置1：无回退层，用来测试性能
# disabled_names = []

# 配置2：有回退层，用来测试精度
disabled_names = ["transformer.h.0.mlp.c_proj",
"transformer.h.1.attn.c_attn",
"transformer.h.1.mlp.c_fc",
"transformer.h.1.mlp.c_proj",
"transformer.h.2.attn.c_attn",
"transformer.h.2.mlp.c_proj",
"transformer.h.3.attn.c_attn",
"transformer.h.3.mlp.c_proj",
"transformer.h.4.attn.c_attn",
"transformer.h.4.mlp.c_proj",
"transformer.h.11.attn.c_attn",
"transformer.h.12.mlp.c_fc",
"transformer.h.13.mlp.c_fc",
"transformer.h.14.mlp.c_fc",
"transformer.h.15.mlp.c_fc",
"transformer.h.16.mlp.c_fc", 
"transformer.h.17.mlp.c_fc",
"transformer.h.18.mlp.c_fc",
"transformer.h.19.mlp.c_fc",
"transformer.h.20.mlp.c_fc",
"transformer.h.21.mlp.c_fc",
"transformer.h.39.attn.c_attn",
"transformer.h.39.mlp.c_fc",
"transformer.h.39.mlp.c_proj",
"lm_head"]

# 配置量化参数，并返回量化配置实例
quant_config = QuantConfig(disable_names=disabled_names, w_bit=8, dev_type='cpu', 
                            act_method=3, pr=1.0, mm_tensor=False)
# 输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')

#执行量化
calibrator.run() 

# 可选
# inference(model, tokenizer)

# calibrator.save('/home/data/starcoder_quant_new', save_type=["numpy", "safe_tensor"]) # save()保存模型量化参数
calibrator.save('/home/data/starcoder_quant_new', save_type=["safe_tensor"]) # save()保存模型量化参数
print(f"saved successfully")