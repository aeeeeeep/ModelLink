import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent
TEST_DIR = os.path.join(BASE_DIR, 'tests')


# gpt test
shell_file = os.path.join(TEST_DIR, "st", "test_gpt", "test_gpt_ptd.sh")
os.system("sh {}".format(shell_file))

# llama test
shell_file = os.path.join(TEST_DIR, "st", "test_llama", "test_llama_ptd.sh")
os.system("sh {}".format(shell_file))

if os.path.exits("/usr1/workspace/PyTorch_PR_AscendSpeed_master/CODE/log.npy"):
    sys.exit(1)
else:
    sys.exit(0)