import os
import logging
import time
import shutil
import torch


prompts = [
    "Common sense questions\n\nQuestion:What is a banana?",
]


def init_resource(model_config):
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    logging.basicConfig(format='%(asctime)s - %(levelname)s - [LLM](%(filename)s:%(lineno)d): %(message)s',
                        level=logging.INFO)
    logging.getLogger("paramiko").setLevel(logging.ERROR)

    device_num = len(model_config["device_list"])
    os.environ["DEVICE_NUM"] = str(device_num)
    os.environ["INPUT_PADDING"] = str(model_config["input_padding"])
    os.environ["BACKEND"] = model_config["backend"]


def set_options(model_config):
    backend = model_config["backend"]
    exe_mode = model_config["exe_mode"]
    jit_compile = model_config["jit_compile"]

    if backend == "npu":
        torch.npu.set_compile_mode(jit_compile=jit_compile)
        npu_options = {"NPU_FUZZY_COMPILE_BLACKLIST": "ReduceProd"}
        torch.npu.set_option(npu_options)

    os.environ["EXE_MODE"] = exe_mode


def generate_prompts(model_config):
    batch_prompts = []
    batch_prompts.append(prompts)
    model_config["batch_prompts"] = batch_prompts


def generate_params(inputs, model_config):
    seq_len_out = model_config["seq_len_out"]
    device = model_config["device"]
    kwargs_params = {"max_new_tokens": seq_len_out}
    for key in inputs.keys():
        kwargs_params.update({
            key:inputs[key].to(device)
        })
    return kwargs_params


def generate_answer(model_config):
    model = model_config["model"]
    tokenizer = model_config["tokenizer"]
    seq_len_in = model_config["seq_len_in"]
    device_list = model_config["device_list"]
    local_rank = model_config["local_rank"]
    batch_prompts = model_config["batch_prompts"]
    input_padding = model_config["input_padding"]

    warmup_prompts = batch_prompts[0]
    if input_padding:
        warmup_inputs = tokenizer(warmup_prompts,
                                  return_tensors="pt", # 返回pytorch tensor
                                  truncation=True,
                                  padding='max_length',
                                  max_length=seq_len_in)
    else:
        warmup_inputs = tokenizer(warmup_prompts,
                                  return_tensors="pt", # 返回pytorch tensor
                                  truncation=True)

    kwargs_params = generate_params(warmup_inputs, model_config)
    start_time = time.time()
    with torch.no_grad():
        _ = model.generate(**kwargs_params)
    elapse = time.time() - start_time
    is_logging = (len(device_list) > 1 and (local_rank == 0 or local_rank == device_list[0])) or (len(device_list) == 1)
    if is_logging:
        logging.info("Execution of warmup is finished, time cost: %.2fs", elapse)

    # execute inference
    for prompt in batch_prompts:
        if input_padding:
            inputs = tokenizer(prompt,
                               return_tensors="pt", # 返回pytorch tensor
                               truncation=True,
                               padding='max_length',
                               max_length=seq_len_in)
        else:
            inputs = tokenizer(prompt,
                               return_tensors="pt", # 返回pytorch tensor
                               truncation=True)
    kwargs_params = generate_params(inputs, model_config)
    start_time = time.time()
    with torch.no_grad():
        generate_ids = model.generate(**kwargs_params)
    elapse = time.time() - start_time

    input_tokens = len(inputs.input_ids[0])
    output_tokens = len(generate_ids[0])
    new_tokens = output_tokens - input_tokens
    res = tokenizer.batch_decode(generate_ids[:, input_tokens:],
                                  skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)

    if is_logging:
        logging.info("E2E inference is finished, time cost:%.2fs", elapse)
        if isinstance(res, list):
            for answer in res:
                logging.info("Inference decode result: \n%s", answer)
        else:
            logging.info("Inference decode result: \n%s", res)
        logging.info("Output tokens number: %s, input tokens number:%s, total new tokens generated: %s",
                     output_tokens, input_tokens, new_tokens)
