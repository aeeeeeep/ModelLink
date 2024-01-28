import os
import time

import torch
import torch_npu
import transformers
from transformers import AutoTokenizer

# 修改model的引入路径
from configuration_gpt_neox import GPTNeoXConfig
from patches.models.modeling_gpt_neox_model import GPTNeoXForCausalLM

# 设置跑的卡
DEVICE_ID = os.environ.get("SET_NPU_DEVICE")
device_id = 0
if DEVICE_ID is not None:
    device_id = int(DEVICE_ID)
print(f"[WARNING] USE npu:{device_id}")
torch.npu.set_device(torch.device(f"npu:{device_id}"))

# 打开算子二进制编译
torch.npu.set_compile_mode(jit_compile=False)
option = {}
option["NPU_FUZZY_COMPILE_BLACKLIST"] = "ReduceProd"
torch.npu.set_option(option)
model_path = "/data/acltransformer_testdata/weights/gptneox20b/"


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


def warm_up(model):
    past_key_values = None
    dummy_input_ids_full = torch.randint(0, 32000, [1, 128], dtype=torch.long).npu()
    dummy_position_ids_full = torch.arange(0, 128, dtype=torch.long).repeat(1, 1).npu()
    dummy_position_ids_full = dummy_position_ids_full.view(-1, 128)
    model_inputs = {
        "input_ids": dummy_input_ids_full,
        "past_key_values": None,
        "position_ids": dummy_position_ids_full,
    }
    with torch.no_grad():
        out = model(**model_inputs,
                    return_dict=True,
                    output_attentions=False,
                    output_hidden_states=False,
                    use_cache=True)
    dummy_input_ids_nxt = torch.randint(0, 32000, [1, 1], dtype=torch.long).npu()
    dummy_past_key_values = out.past_key_values
    seq_length = dummy_input_ids_nxt.shape[1]
    pkv_length = dummy_past_key_values[0][0].shape[2]
    position_ids = torch.arange(seq_length, 2048, dtype=torch.long).repeat(1, 1).npu()
    dummy_position_ids_nxt = position_ids.view(-1, seq_length)
    for _ in range(5):
        past_key_values = out.past_key_values
        input_ids = torch.randint(32000, (1, 1)).npu()
        position_ids = torch.randint(2048, (1, 1)).npu()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            # "attention_mask": dummy_attention_mask_nxt
        }
        with torch.no_grad():
            out = model(**model_inputs,
                        return_dict=True,
                        output_attentions=False,
                        output_hidden_states=False,
                        use_cache=True)


def full_and_incremental_test(seq_len, batch, test_cycle, model, tokenizer):
    print("start run.")
    warm_up(model)
    past_key_values = None
    dummy_input_ids_full = torch.randint(0, len(tokenizer), [batch, seq_len], dtype=torch.long).npu()
    dummy_position_ids_full = torch.arange(0, seq_len, dtype=torch.long).repeat(batch, 1).npu()
    dummy_position_ids_full = dummy_position_ids_full.view(-1, seq_len)
    model_inputs = {
        "input_ids": dummy_input_ids_full,
        "past_key_values": None,
        "position_ids": dummy_position_ids_full,
    }
    torch.npu.synchronize()
    start = time.time()
    with torch.no_grad():
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
    dummy_input_ids_nxt = torch.randint(0, len(tokenizer), [batch, 1], dtype=torch.long).npu()
    dummy_past_key_values = out.past_key_values
    seq_length = dummy_input_ids_nxt.shape[1]
    pkv_length = dummy_past_key_values[0][0].shape[2]
    position_ids = torch.arange(seq_length, 2048, dtype=torch.long).repeat(batch, 1).npu()
    dummy_position_ids_nxt = position_ids.view(-1, seq_length)
    for i in range(test_cycle):
        past_key_values = out.past_key_values
        input_ids = torch.randint(len(tokenizer), (batch, 1)).npu()
        position_ids = torch.randint(2048, (batch, 1)).npu()
        model_inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "position_ids": position_ids,
            # "attention_mask": dummy_attention_mask_nxt
        }
        torch.npu.synchronize()
        start = time.time()
        with torch.no_grad():
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

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model_config = GPTNeoXConfig.from_pretrained(model_path)
    model = GPTNeoXForCausalLM.from_pretrained(model_path, trust_remote_code=True,
                                               config=model_config).half().npu()

    file = open(f"zhiputest_{device_version}_gptneox_20b.csv", 'w')
    file.write(
        f"Batch,MaxSeqLen,InputSeqLen(Encoding),OutputSeqLen(Decoding),TokensPerSecond(ms),ResponseTime(ms),FirstTokenTime(ms),TimePerTokens(ms)\n")
    for batch_level in [1]:
        for seq_len_level in range(5, 11):
            for test_cycle_level in range(5, 11):
                seq_len = 2 ** seq_len_level
                test_cycle = 2 ** test_cycle_level
                input_param = {"seq_len": seq_len,
                               "batch": batch_level,
                               "test_cycle": test_cycle,
                               "model": model,
                               "tokenizer": tokenizer}
                print(f"batch: {batch_level}, seq_len: {seq_len}, test_cycle: {test_cycle}")
                first_time, avg_token = full_and_incremental_test(**input_param)
                file.write(
                    f"{batch_level},2048,{seq_len},{test_cycle},{round(1000 / avg_token, 2)},{round(first_time + avg_token * test_cycle, 2)},{round(first_time, 2)},{round(avg_token, 2)}\n")

    file.close()
