#  baichuan2-7B Model-Single core inference guidance


# Overview

Baichuan 2 is the new generation of large-scale open-source language models launched by Baichuan Intelligence inc.. It is trained on a high-quality corpus with 2.6 trillion tokens and has achieved the best performance in authoritative Chinese and English benchmarks of the same size. This release includes 7B and 13B versions for both Base and Chat models, along with a 4bits quantized version for the Chat model. All versions are fully open to academic research, and developers can also use them for free in commercial applications after obtaining an official commercial license through email request. 

- Reference implementation：

  ```
  https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat
  ```

# Input and output data

- Input data

  | Input data       | File size                               | Data type | Data layout format | Required |
  | -------------- | ---------------------------------- | -------- | ------------ | -------- |
  | input_ids      | BATCH_SIZE x SEQ_LEN               | INT64    | ND           | Yes       |
  | attention_mask | BATCH_SIZE x 1 x SEQ_LEN x SEQ_LEN | FLOAT32  | ND           | No        |

- Output data

  | Output data   | File size                        | Data type | Data layout format |
  | ---------- | --------------------------- | -------- | ------------ |
  | output_ids | BATCH_SIZE x OUTPUT_SEQ_LEN | INT64    | ND           |

# Reasoning environment preparation

- This model requires the following plug-ins and drivers

  **Table 1** Version matching table

  | Name           | Version          | Download link |
  | -------------- | ------------- | -------- |
  | Firmware and drivers     | 23.0.RC3.B060 | -        |
  | CANN           | 7.0.RC1.B060  | -        |
  | Python         | 3.9.11        | -        |
  | PytorchAdapter | 1.11.0        | -        |
  | Inference engine       | -             | -        |

  **Table 2** Inference engine dependencies

  | Software  | Version |
  | ----- | -------- |
  | glibc | >= 2.27  |
  | gcc   | >= 7.5.0 |

  **Table 3** Hardware

  | CPU     | Device |
  | ------- | ------ |
  | aarch64 | 910B3   |
  | x86     | 910B3   |

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
    File path：/Alpha/Ascend910B
    Installation method：
    | Package names                                          |
    | --------------------------------------------------- |
    | Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run |
    ```bash
    # install firmwire
    chmod +x Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run
    ./Ascend-hdk-910b-npu-firmware_7.0.t9.0.b221.run --full
    ```

  - driver\
    File path：/Alpha/Ascend910B
    Installation method：
    | cpu     | Package names                                          |
    | ------- | --------------------------------------------------- |
    | aarch64 | Ascend-hdk-910b-npu-driver_23.0.rc3.b060_linux-aarch64.run |
    | x86     | Ascend-hdk-910b-npu-driver_23.0.rc3.b060_linux-x86_64.run  |

    ```bash
    # install driver
    chmod +x Ascend-hdk-910b-npu-driver_23.0.rc3.b060_*.run
    ./Ascend-hdk-910b-npu-driver_23.0.rc3.b060_*.run --full
    ```
  
- Install CANN
  Install toolkit first,then install kernel
  Download link：
  ```
  https://cmc-szv.clouddragon.huawei.com/cmcversion/index/releaseView?deltaId=8685077595423104&isSelect=Software 
  ```
  - toolkit\
    File path：/run/aarch64-linux
    Installation method：
    | Package names                                          |
    | --------------------------------------------------- |
    | Ascend-cann-toolkit_7.0.T10_linux-aarch64.run |
    ```bash
    # install toolkit, arm for example
    chmod +x Ascend-cann-toolkit_7.0.T10_linux-aarch64.run
    ./Ascend-cann-toolkit_7.0.T10_linux-aarch64.run --install
    source /usr/local/Ascend/ascend-toolkit/set_env.sh
    ```

  - kernel\
    File path：/run
    Installation method：
    | Package names                                          |
    | --------------------------------------------------- |
    | Ascend-cann-kernels-910b_7.0.T10_linux.run |

    ```bash
    # 安装 kernel
    chmod +x Ascend-cann-kernels-910b_7.0.T10_linux.run
    ./Ascend-cann-kernels-910b_7.0.T10_linux.run --install
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
    | Package names                                          |
    | --------------------------------------------------- |
    | torch-1.11.0+cpu-cp38-cp38-linux_x86_64.whl |
    | torch-1.11.0+cpu-cp39-cp39-linux_x86_64.whl |
    | torch-1.11.0+cpu-cp310-cp310-linux_x86_64.whl |
    | ... |
    Select version torch-1.11.0.

    ```bash
    # isntall torch , arm for example
    pip install torch-1.11.0-cp38-cp38-linux_aarch64.whl
    ```

  - torch_npu\
    File path：/
    Installation method：
    | Package names                                          |
    | --------------------------------------------------- |
    | pytorch_v1.11.0_py38.tar.gz |
    | ... |
	Please select the same version with torch

    ```bash
    # intsll  torch_npu , arm for example
    tar -zxvf pytorch_v1.11.0_py38.tar.gz
    pip install torch*_aarch64.whl
    ```

- Install dependencies

  Reasoning environment preparation

1. Download baichuan2-7b weights，put them in `input_dir`

   ```
   https://huggingface.co/baichuan-inc/Baichuan2-7B-Chat/tree/main
   ```

2. Install the acceleration library according to the version release link

   | Acceleration package names                                          |
   | --------------------------------------------------- |
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

   | cpu     | LLM package names                                |
   | ------- | ----------------------------------------- |
   | aarch64 | Ascend-cann-llm_{version_id}_linux-aarch64_torch{pta_version}-abi0.tar.gz  |
   | x86     | Ascend-cann-llm_{version_id}_linux-x86_64_torch{pta_version}-abi0.tar.gz |
   
   Whether to use cxx11abi0 or cxx11abi1 is the same as installing atb.

   ```bash
   # install large language model
   # cd {baichuan2_path}
   tar -xzvf Ascend-cann-llm_*.tar.gz
   source set_env.sh
   ```

   > Note: Source CANN, acceleration library, and large model are required before each run.

## Model inference

- Copy Files
  1. Copy files `config.json` 、`configuration_baichuan.py` 、`generation_utils.py`  、`quantizer.py` `tokenization_baichuan.py` 、`tokenizer_config.json` to {baichuan2_path}/pytorch/examples/baichuan2/7b，example：
  ```
  cd {baichuan2_path}/pytorch/examples/baichuan2/7b
  cp /data/models/baichuan2/7b/config.json ./
  ...
  ```
- edit `config.json` \
  change key `auto_map.AutoModelForCausalLM` 's value to "modeling_baichuan_ascend.BaichuanForCausalLM"
     
  
- Soft link model weight file

  ```
  ln -s {model_path}/pytorch_model.bin  pytorch_model.bin
  ln -s {model_path}/tokenizer.model  tokenizer.model
  ```

- Perform inference

  ```
  python run_baichuan_half.py
  ```

  This command will run a simple inference instance warm up and start a subsequent question and answer session.

- Customized operation can refer to `run_baichuan_half.py`

# Model inference performance 

  | Hardware type | Input length | Output length | Decode speed      |
  | --------      | -------- | -------- | ------------- |
  | 310P Parallel  | 1 x 1024 | 1024     | 10.41 tokens/s |
  | 310P Parallel  | 1 x 2048 | 2048     | 7.42 tokens/s |
  | 910B      |  1 x 1024 | 1024     | 35.69 tokens/s |
  | 910B      |  1 x 2048 | 2048     | 32.19 tokens/s |