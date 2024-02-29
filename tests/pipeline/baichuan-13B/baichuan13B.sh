#!/bin/bash
# Provide uniform access for piepline.

python tests/pipeline/baichuan-13B/test_process_pretrain_data.py
python tests/pipeline/baichuan-13B/test_process_instruction_data.py
pytest -s tests/pipeline/baichuan-13B/test_generation.py
pytest -s tests/pipeline/baichuan-13B/test_evaluation.py
pytest -s tests/pipeline/baichuan-13B/test_lora.py
pytest -s tests/pipeline/baichuan-13B/test_trainer.py


