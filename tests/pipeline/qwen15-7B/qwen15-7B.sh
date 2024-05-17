# Provide uniform access for piepline.

python tests/pipeline/qwen15-7B/test_convert_ckpt_from_huggingface.py

pytest -s tests/pipeline/qwen15-7B/test_generation.py
pytest -s tests/pipeline/qwen15-7B/test_evaluation.py
