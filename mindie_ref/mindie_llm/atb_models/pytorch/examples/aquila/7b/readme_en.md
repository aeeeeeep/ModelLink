#  Aquila-7B Model-Single core inference guidance


# Overview

Aquila Language Model is the first open source language model that supports both Chinese and English knowledge, commercial license agreements, and compliance with domestic data regulations.

- Reference implementation：

  ```
    https://huggingface.co/BAAI/Aquila-7B
  ```


# Reasoning environment preparation

- This model requires the following plug-ins and drivers

  **Table 1** Version matching table

  | Name                 | Version      | Download link |
  |----------------------|--------------|---------------|
  | Firmware and drivers | 24.0.T1      | -             |
  | CANN                 | 8.0.T2.B010  | -             |
  | Python               | 3.9.18       | -             |
  | PytorchAdapter       | 6.0.RC1.B011 | -             |

  **Table 2** Inference engine dependencies

  | Software | Version  |
  |----------|----------|
  | glibc    | >= 2.27  |
  | gcc      | >= 7.5.0 |

  **Table 3** Hardware

  | CPU     | Device         |
  |---------|----------------|
  | aarch64 | Atlas 800I A2  |
  | aarch64 | Atlas 300I DUO |

# Quickly start

## Get source code and dependencies

### 1. Environment deployment

- Install HDK  
  Install firmwire first，then install driver
  Download link：
  ```
  https://cmc.rnd.huawei.com/cmcversion/index/releaseView?deltaId=8724961914913280&isSelect=Software 
  ```
  - firmwire
    Installation method：
   
  - | Package names                        |
    |--------------------------------------|
    | Ascend-hdk-xxxx-npu-firmware_xxx.run |
    ```bash
    # install firmwire
    chmod +x Ascend-hdk-xxxx-npu-firmware_xxx.run
    ./Ascend-hdk-xxxx-npu-firmware_xxx.run --full
    ```

  - driver\
    Installation method：
    
  - | cpu     | Package names                                    |
    |---------|--------------------------------------------------|
    | aarch64 | Ascend-hdk-xxxx-npu-driver_xxx_linux-aarch64.run |
    | x86     | Ascend-hdk-xxxx-npu-driver_xxx_linux-x86_64.run  |

    ```bash
    # install driver
    chmod +x Ascend-hdk-xxxx-npu-driver_xxx_*.run
    ./Ascend-hdk-xxxx-npu-driver_xxx_*.run --full
    ```
  
- Install CANN
  Install toolkit first,then install kernel
  Download link：
  ```
  https://cmc-szv.clouddragon.huawei.com/cmcversion/index/releaseView?deltaId=8685077595423104&isSelect=Software 
  ```
  - toolkit\
    Installation method：
    
  - | Package names                                 |
    |-----------------------------------------------|
    | Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run |
    ```bash
    # install toolkit, arm for example
    chmod +x Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run
    ./Ascend-cann-toolkit_8.0.RC1_linux-aarch64.run --install
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

  - kernel\
    File path：/run
    Installation method：
   
  - | Package names                              |
    |--------------------------------------------|
    | Ascend-cann-kernels-xxxx_8.0.RC1_linux.run |

    ```bash
    # 安装 kernel
    chmod +x Ascend-cann-kernels-xxxx_8.0.RC1_linux.run
    ./Ascend-cann-kernels-xxxx_8.0.RC1_linux.run --install
    ```

- Install PytorchAdapter
  Install torch first, then install torch_npu
  Download link：
  ```
  https://cmc-szv.clouddragon.huawei.com/cmcversion/index/releaseView?deltaId=8865172195444096&isSelect=Inner 
  ```
  - torch\
    File path：/torch
    Installation method：
   
  - | Package names                                   |
    |-------------------------------------------------|
    | torch-2.0.1+cpu-cp38-cp38-linux_x86_64.whl      |
    | torch-2.0.1+cpu-cp39-cp39-linux_x86_64.whl      |
    | torch-2.0.1-cp38-cp38-manylinux2014_aarch64.whl |
    | torch-2.0.1-cp39-cp39-manylinux2014_aarch64.whl |
    | ...                                             |
    Select version torch-2.0.1.

    ```bash
    # isntall torch , arm for example
    pip install torch-2.0.1-cp39-cp39-linux_aarch64.whl
    ```

  - torch_npu\
    File path：/
    Installation method：
    
  - | Package names              |
    |----------------------------|
    | pytorch_v2.0.1_py38.tar.gz |
    | pytorch_v2.0.1_py39.tar.gz |
    | ...                        |
	Please select the same version with torch

    ```bash
    # intsll  torch_npu , arm for example
    tar -zxvf pytorch_v2.0.1_py39.tar.gz
    pip install torch*_aarch64.whl
    ```

- Install dependencies

  Reasoning environment preparation

1. Download Aquila-7b weights，put them in `${model_download_path}`

   ```
   https://huggingface.co/BAAI/Aquila-7B/tree/main
   ```

2. Install the acceleration library according to the version release link

   | Acceleration package names                            |
   |-------------------------------------------------------|
   | Ascend-cann-atb_{version}_cxx11abi0_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-aarch64.run |
   | Ascend-cann-atb_{version}_cxx11abi1_linux-x86_64.run  |
   | Ascend-cann-atb_{version}_cxx11abi0_linux-x86_64.run  |
   
   Whether to use cxx11abi0 or cxx11abi1 can be queried through the python command
   ```
   import torch
   torch.compiled_with_cxx11_abi()
   ```
   If True is returned, cxx11abi1 is used, otherwise use cxx11abi0

   ```bash
   # install atb
   chmod +x Ascend-cann-atb_*.run
   ./Ascend-cann-atb_*.run --install
   source /usr/local/Ascend/atb/set_env.sh
   ```

3. Unzip the large language model file according to the version release link

   | cpu     | LLM package names                                                         |
   |---------|---------------------------------------------------------------------------|
   | aarch64 | Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz |
   | x86     | Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz  |
   
   Whether to use cxx11abi0 or cxx11abi1 is the same as installing atb.

   ```bash
   # install large language model
   # cd {llm_path}
   tar -xzvf Ascend-cann-llm_*.tar.gz
   source set_env.sh
   ```

   > Note: Source CANN, acceleration library, and large model are required before each run.

## Model inference

- Copy Files
  1. Copy files `config.json` 、`configuration_aquila.py` 、`generation_utils.py` 、`tokenizer_config.json` to `{model_path}`，example：
  ```shell
  cp ${model_download_path}/*.py ${model_path}/
  cp ${model_download_path}/*.json ${model_path}/
  cp ${model_download_path}/*.model ${model_path}/
  ln -s ${model_download_path}/*.bin ${model_path}/
  ```
  
- edit `config.json` \
  change key `AutoModelForCausalLM` 's value to "modeling_aquila_ascend.AquilaForCausalLM"
     
  
- Soft link model weight file

  ```
  ln -s {model_path}/pytorch_model.bin  pytorch_model.bin
  ```

- Perform inference

  ```
  python main.py
  ```

  This command will run a simple inference instance warm up and start a subsequent question and answer session.

- Customized operation can refer to `main.py`

# Model inference performance 

# Atlas 800I A2

## precision

| precision      | Atlas 800I A2       | A100                | compare              |
|----------------|---------------------|---------------------|----------------------| 
| STEM           | 0.3767441860465116  | 0.3813953488372093  | -0.0046511627906977  |
| Social Science | 0.48363636363636364 | 0.48363636363636364 | 0                    |
| Humanities     | 0.41245136186770426 | 0.41245136186770426 | 0                    |
| Other          | 0.3958333333333333  | 0.3932291666666667  | +0.0026041666666666  |
| Avg acc        | 0.41084695393759285 | 0.4115898959881129  | -0.00074294205052005 |

## performance

| Device        | batch_size | forward_first_token_speed(token/s) | forward_next_token_speed(token/s) |
|---------------|------------|------------------------------------|-----------------------------------|
| A100          | 1          | 22.68088002                        | 88.88888889                       |
| Atlas 800I A2 | 1          | 24.20406347                        | 67.56546904                       |
| compare       | 1          | 1.067157158                        | 0.760111527                       |


# Atlas 300I DUO

## precision

| precision      | Atlas 300I DUO      | 
|----------------|---------------------|
| STEM           | 0.3627906976744186  |
| Social Science | 0.49818181818181817 |
| Humanities     | 0.43190661478599224 |
| Other          | 0.4088541666666667  |
| Avg acc        | 0.41679049034175336 |

## performance


| Device         | batch_size | input_seq_len(Encoding) | output_seq_len(Decoding) | forward_first_token_time(ms) | forward_next_token_time(ms) |
|----------------|------------|-------------------------|--------------------------|------------------------------|-----------------------------|
| Atlas 300I DUO | 1          | 2^5~2^10                | 2^5~2^10                 | 235.0490424                  | 91.39817598                 |

