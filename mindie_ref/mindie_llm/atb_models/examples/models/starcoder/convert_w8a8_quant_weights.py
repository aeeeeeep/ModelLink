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
calib_list = [
    "def print_hello_world():",
    "def Fibonacci_sequence(n):",
]

#校准数据获取
dataset_calib = get_calib_dataset(tokenizer, calib_list) 

# 量化配置
# 配置回退层数
disabled_names = []
disabled_layers = [0, 1, 2, 3, 39]
for i in disabled_layers:
    disabled_names.append(f"transformer.h.{i}.attn.c_attn")
    disabled_names.append(f"transformer.h.{i}.attn.c_proj")
    disabled_names.append(f"transformer.h.{i}.mlp.c_fc")
    disabled_names.append(f"transformer.h.{i}.mlp.c_proj")
# 配置量化参数，并返回量化配置实例
quant_config = QuantConfig(disable_names=disabled_names, w_bit=8, dev_type='cpu', 
                            act_method=3, pr=0.5, mm_tensor=False)
# 输入加载的原模型、量化配置和校准数据，定义校准
calibrator = Calibrator(model, quant_config, calib_data=dataset_calib, disable_level='L0')

#执行量化
calibrator.run() 

# 可选
# inference(model, tokenizer)

# calibrator.save('/home/data/starcoder_quant_new', save_type=["numpy", "safe_tensor"]) # save()保存模型量化参数
calibrator.save('/home/data/starcoder_quant_new', save_type=["safe_tensor"]) # save()保存模型量化参数
print(f"saved successfully")