# Provide uniform access for piepline.

python tests/pipeline/llama3-8B/test_convert_ckpt_from_huggingface.py

pytest -s tests/pipeline/llama3-8B/test_generation.py
pytest -s tests/pipeline/llama3-8B/test_evaluation.py