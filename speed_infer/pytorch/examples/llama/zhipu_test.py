import torch
import time
import os
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch_npu
from torch_npu.contrib import transfer_to_npu
import argparse

# 设置跑的卡
DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
device_id = 6
if DEVICE_ID is not None:
    device_id = int(DEVICE_ID)
print(f"[WARNING] USE npu:{device_id}")
torch.npu.set_device(torch.device(f"npu:{device_id}"))

# 打开算子二进制编译
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)

# 修改transformers的TopPLogitsWarper
def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor) -> torch.FloatTensor:
    sorted_logits, sorted_indices = torch.sort(scores, descending=False)
    # cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
    cumulative_probs = sorted_logits.softmax(
        dim=-1).cpu().float().cumsum(dim=-1).to(sorted_logits.device).to(sorted_logits.dtype)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs <= (1 - self.top_p)
    if self.min_tokens_to_keep > 1:
        # Keep at least min_tokens_to_keep
        sorted_indices_to_remove[..., -self.min_tokens_to_keep:] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(
        1, sorted_indices, sorted_indices_to_remove)
    scores = scores.masked_fill(indices_to_remove, self.filter_value)
    return scores

transformers.generation.TopPLogitsWarper.__call__ = __call__

def nd_nz(model):
    soc_version = torch_npu._C._npu_get_soc_version()
    if soc_version in [104, 220, 221, 222, 223]:
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.npu_format_cast(2)
        print("soc_version:", soc_version, " is 910B, support ND")
    else:
        for name, module in model.named_modules():
            # 如果是lm_head把weights存下来，不走NZ
            if isinstance(module, torch.nn.Linear):
                module.weight.data = module.weight.data.npu_format_cast(29)
        print("soc_version:", soc_version, " is not 910B, support NZ")
        
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            module.weight.data = module.weight.data.npu_format_cast(2)
            

def warm_up(model):
    past_key_values = None
    dummy_input_ids_full = torch.randint(0, 32000, [1, 128], dtype=torch.int).npu()
    dummy_position_ids_full = torch.arange(0, 128, dtype=torch.long).repeat(1, 1).npu()
    dummy_position_ids_full = dummy_position_ids_full.view(-1, 128)
    model_inputs = {
    "input_ids": dummy_input_ids_full,
    "past_key_values": None,
    "position_ids": dummy_position_ids_full,
    }
    out = model(**model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False, 
            use_cache=True)
    dummy_input_ids_nxt = torch.randint(0, 32000, [1, 1], dtype=torch.int).npu()
    dummy_past_key_values = out.past_key_values
    seq_length = dummy_input_ids_nxt.shape[1]
    pkv_length = dummy_past_key_values[0][0].shape[2]
    position_ids = torch.arange(pkv_length, seq_length + pkv_length, dtype=torch.long).repeat(1, 1).npu()
    dummy_position_ids_nxt = position_ids.view(-1, seq_length)
    for _ in range(5):
        past_key_values = out.past_key_values
        input_ids = torch.randint(150000, (1, 1)).npu()
        position_ids = torch.randint(2048, (1, 2, 1)).npu()
        model_inputs = {
        "input_ids": dummy_input_ids_nxt,
        "past_key_values": dummy_past_key_values,
        "position_ids": dummy_position_ids_nxt,
        # "attention_mask": dummy_attention_mask_nxt
        }
        out = model(**model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True)

def full_and_incremental_test(seq_len, batch, test_cycle, model):
    print("start run.")
    warm_up(model)
    past_key_values = None
    dummy_input_ids_full = torch.randint(0, 32000, [batch, seq_len], dtype=torch.int).npu()
    dummy_position_ids_full = torch.arange(0, seq_len, dtype=torch.long).repeat(batch, 1).npu()
    dummy_position_ids_full = dummy_position_ids_full.view(-1, seq_len)
    model_inputs = {
    "input_ids": dummy_input_ids_full,
    "past_key_values": None,
    "position_ids": dummy_position_ids_full,
    }
    torch.npu.synchronize()
    start = time.time()
    out = model(**model_inputs,
            return_dict=True,
            output_attentions=False,
            output_hidden_states=False, 
            use_cache=True)
    
    torch.npu.synchronize()
    end = time.time()
    first_time = (end - start) * 1000
    print(f"first token: {first_time}ms")
    sum_time = 0
    test_cycle -= 1
    dummy_input_ids_nxt = torch.randint(0, 32000, [batch, 1], dtype=torch.int).npu()
    dummy_past_key_values = out.past_key_values
    seq_length = dummy_input_ids_nxt.shape[1]
    pkv_length = dummy_past_key_values[0][0].shape[2]
    position_ids = torch.arange(pkv_length, seq_length + pkv_length, dtype=torch.long).repeat(batch, 1).npu()
    dummy_position_ids_nxt = position_ids.view(-1, seq_length)
    for i in range(test_cycle):
        past_key_values = out.past_key_values
        input_ids = torch.randint(150000, (batch, 1)).npu()
        position_ids = torch.randint(2048, (1, 2, 1)).npu()
        model_inputs = {
        "input_ids": dummy_input_ids_nxt,
        "past_key_values": dummy_past_key_values,
        "position_ids": dummy_position_ids_nxt,
        # "attention_mask": dummy_attention_mask_nxt
        }
        torch.npu.synchronize()
        start = time.time()
        out = model(**model_inputs,
                return_dict=True,
                output_attentions=False,
                output_hidden_states=False,
                use_cache=True)
        torch.npu.synchronize()
        end = time.time()
        cur_time = (end - start) * 1000
        sum_time += cur_time
        dummy_past_key_values = out.past_key_values
    avg_time = sum_time / test_cycle
    print(f"average token: {sum_time / test_cycle}ms")
    print(f"response time: {first_time + sum_time}ms")
    return first_time, avg_time

if __name__ == "__main__":
    soc_version_map = {-1: "unknown soc version",
    100: "910PremiumA", 101: "910ProA", 102: "910A", 103: "910ProB", 104: "910B",
    200: "310P1", 201: "310P2", 202: "310P3", 203: "310P4",
    220: "910B1", 221: "910B2", 222: "910B3", 223: "910B4",
    240: "310B1", 241: "310B2", 242: "310B3",
    250: "910C1", 251: "910C2", 252: "910C3", 253: "910C4"
    }
    device_version = soc_version_map[torch_npu._C._npu_get_soc_version()]
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        help="MODEL SELECT [llama1-7b|llama1-13b|llama2-7b|llama2-13b]",
    )
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("./", trust_remote_code=True, use_fast=False)
    model = AutoModelForCausalLM.from_pretrained("./", trust_remote_code=True).half().npu()
    nd_nz(model)
    file = open(f"zhiputest_{device_version}_{args.model_name}.csv", 'w')
    file.write(f"Batch,max_seq_token,input_seq_len(Encoding),output_seq_len(Decoding),TokensPerSecond(ms),ResponseTime(ms),FirstTokenTime(ms),TimePerTokens(ms)\n")
    for batch_level in [1]:
        for seq_len_level in range(5, 11):
            for test_cycle_level in range(5, 11):
                seq_len = 2 ** seq_len_level
                test_cycle = 2 ** test_cycle_level
                input_param = {"seq_len": seq_len,
                            "batch": batch_level,
                            "test_cycle": test_cycle,
                            "model": model}
                print(f"batch: {batch_level}, seq_len: {seq_len}, test_cycle: {test_cycle}")
                first_time, avg_token = full_and_incremental_test(**input_param)
                file.write(f"{batch_level},2048,{seq_len},{test_cycle},{round(1000/avg_token,2)},{round(first_time+avg_token*test_cycle, 2)},{round(first_time, 2)},{round(avg_token, 2)}\n")

    file.close()