#!/bin/bash
# export LD_PRELOAD=/usr/lib64/libgomp.so.1.0.0:$LD_PRELOAD

# ATB_LOG_TO_STDOUT=1 ATB_LOG_LEVEL=INFO TASK_QUEUE_ENABLE=0 ASDOPS_LOG_TO_STDOUT=1 ASDOPS_LOG_LEVEL=INFO
#coco
python run_ascend_coco.py with "task_finetune_irtr_coco_base_image384"  load_path="/data1/models/vlmo/vlmo_base_patch16_384_coco.pt" test_only=True
