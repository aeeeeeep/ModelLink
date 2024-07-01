# Provide uniform access for piepline.

python tests/pipeline/gemma-7B/test_process_pretrain_data.py
python tests/pipeline/gemma-7B/test_convert_ckpt_from_huggingface.py

pytest -s tests/pipeline/gemma-7B/test_generation.py
pytest -s tests/pipeline/gemma-7B/test_evaluation.py
