# Provide uniform access for piepline.

python tests/pipeline/intern-7B/test_process_pretrain_data.py

pytest -s tests/pipeline/intern-7B/test_generation.py
pytest -s tests/pipeline/intern-7B/test_evalution.py
pytest -s tests/pipeline/intern-7B/test_trainer.py
