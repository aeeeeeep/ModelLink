#!/bin/bash

SCRIPT_PATH=./tools/ckpt_convert/bloom/convert_weights_from_ascendspeed_to_hugggingface.py
python $SCRIPT_PATH \
    --output-huggingface-model-dir "your huggingface checkpoint output path" \
    --ascendspeed-model-dir "your ascendspeed checkpoint path" \
    --tensor-model-parallel-size 8 \
    --pipeline-model-parallel-size 12 \
    --type 176B \
    --deepspeed \
    --partition-layers 6,6,6,6,6,6,6,6,6,6,6,4