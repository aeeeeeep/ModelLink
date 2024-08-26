# Provide uniform access for piepline.

python tests/pipeline/glm4-9B/test_convert_ckpt_from_huggingface.py

pytest -s tests/pipeline/glm4-9B/test_generation.py
pytest -s tests/pipeline/glm4-9B/test_evaluation.py