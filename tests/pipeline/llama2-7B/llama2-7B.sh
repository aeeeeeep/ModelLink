# Provide uniform access for piepline.

python tests/pipeline/llama2-7B/test_process_pretrain_data.py
python tests/pipeline/llama2-7B/test_process_instruction_data.py
python tests/pipeline/llama2-7B/test_convert_ckpt_from_huggingface.py

pytest -s tests/pipeline/llama2-7B/test_generation.py
pytest -s tests/pipeline/llama2-7B/test_evaluation.py
pytest -s tests/pipeline/llama2-7B/test_lora.py
pytest -s tests/pipeline/llama2-7B/test_trainer.py
