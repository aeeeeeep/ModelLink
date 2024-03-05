# Provide uniform access for piepline.

python -m unittest test_process_pretrain_data.py
python -m unittest test_convert_ckpt_from_huggingface.py
pytest -s test_generation.py
pytest -s test_evalution.py
pytest -s test_trainer.py
