# Provide uniform access for piepline.
python /tests/pipeline/bloom-7B/test_process_pretrain_data.py
python /tests/pipeline/bloom-7B/test_convert_ckpt_from_huggingface.py
pytest -s /tests/pipeline/bloom-7B/test_trainer.py
pytest -s /tests/pipeline/bloom-7B/test_generation.py
pytest -s /tests/pipeline/bloom-7B/test_evaluation.py
