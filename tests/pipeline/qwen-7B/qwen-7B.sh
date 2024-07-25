# Provide uniform access for piepline.

pytest -s ./tests/pipeline/qwen-7B/test_instruction.py
pytest -s ./tests/pipeline/qwen-7B/test_process_instruction_data.py
pytest -s ./tests/pipeline/qwen-7B/test_generation.py
pytest -s ./tests/pipeline/qwen-7B/test_generation2.py
pytest -s ./tests/pipeline/qwen-7B/test_evaluation.py