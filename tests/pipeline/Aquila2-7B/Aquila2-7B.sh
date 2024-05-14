#!/bin/bash
# Provide uniform access for piepline.

python tests/pipeline/Aquila2-7B/test_convert_weight_from_huggingface.py
pytest -s tests/pipeline/Aquila2-7B/test_generation.py
pytest -s tests/pipeline/Aquila2-7B/test_evaluation.py