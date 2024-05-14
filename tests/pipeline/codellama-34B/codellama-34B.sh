# Provide uniform access for piepline.

python tests/pipeline/codellama-34B/test_convert_ckpt_from_huggingface.py

pytest -s tests/pipeline/codellama-34B/test_generation.py
pytest -s tests/pipeline/codellama-34B/test_evaluation.py