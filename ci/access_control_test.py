import os
import sys
from pathlib import Path

BASE_DIR = Path(__file__).absolute().parent.parent
TEST_DIR = os.path.join(BASE_DIR, 'tests')
flag = False

# gpt test
shell_file = os.path.join(TEST_DIR, "st", "test_gpt", "test_gpt_ptd.sh")
res = os.system("sh {}".format(shell_file))
flag = True if res != 0 else False

# llama test
shell_file = os.path.join(TEST_DIR, "st", "test_llama", "test_llama_ptd.sh")
res = os.system("sh {}".format(shell_file))
flag = True if res != 0 else False


sys.exit(1) if flag else sys.exit(0)