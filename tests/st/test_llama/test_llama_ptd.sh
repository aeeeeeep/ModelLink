export LD_LIBRARY_PATH=/root/miniconda3/lib:$LD_LIBRARY_PATH
export HCCL_CONNECT_TIMEOUT=1200
source /usr/local/Ascend/ascend-toolkit/set_env.sh

pytest ./tests/st/test_llama/test_llama_ptd.py