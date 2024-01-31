TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')/models/telechat

cp modeling_quant_parallel.py $TRANSFORMER_PACKAGE_PATH/modeling_telechat.py

python3 -m torch.distributed.run  --nproc_per_node 2 --master_port 25241 run_parallel.py --runparallel
