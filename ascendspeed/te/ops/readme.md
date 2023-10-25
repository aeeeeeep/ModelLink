# How to run the te ops?

## previous installation
+ Cann
+ Transformer-Boost
+ torch_npu

## compile and install
### 1. set the environment variables

+ export ASCEND_TOOLKIT_HOME = /usr/local/Ascend/latest/
+ export ATB_HOME_PATH = /path/to/ascend-transformer-boost

### 2. include head files

+ newest torch_npu
+ newest cann
+ operation_torch.h in atb
+ hosttensor_binder.h in hosttensor_binders
+ json.hpp in nlohmann

### 3. install scripts
```shell
python3 setup.py build
python3 setup.py bdist
pip3 install dist/*.whl --force-reinstall
```
