# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import json
import os.path
import shutil
import stat


def copy_tokenizer_files(model_dir, dest_dir):
    os.makedirs(dest_dir, exist_ok=True)
    for filename in os.listdir(model_dir):
        if 'tokenizer' in filename or 'tokenization' in filename:
            src_filepath = os.path.join(model_dir, filename)
            dest_filepath = os.path.join(dest_dir, filename)
            shutil.copyfile(src_filepath, dest_filepath)


def modify_config(model_dir, dest_dir, torch_dtype, quantize_type):
    src_config_filepath = os.path.join(model_dir, 'config.json')
    with open(src_config_filepath, 'r', encoding='utf-8') as fr:
        data = json.load(fr)
    data['torch_dtype'] = str(torch_dtype).split(".")[1]
    data['quantize'] = quantize_type
    dest_config_filtpath = os.path.join(dest_dir, 'config.json')
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    models = stat.S_IWUSR | stat.S_IRUSR
    with os.fdopen(os.open(dest_config_filtpath, flags, models), 'w', encoding='utf-8') as fw:
        json.dump(data, fw, indent=4)
