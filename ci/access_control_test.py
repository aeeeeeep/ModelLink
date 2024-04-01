import os
import stat
import sys
import unittest
from pathlib import Path
import xmlrunner


# =============================
# ST test, run with shell
# =============================
def success_check(res):
    if res != 0:
        sys.exit(1)


def success_check_ut(res):
    if len(res.failures) + len(res.errors) != 0:
        sys.exit(1)


class ST_Test:
    
    def __init__(self):

        BASE_DIR = Path(__file__).absolute().parent.parent
        TEST_DIR = os.path.join(BASE_DIR, 'tests')

        st_dir = "st"
        llama_pretrain_shell_file = os.path.join(
            TEST_DIR, st_dir, "test_llama_pretrain_ptd.sh")
        llama_inference_shell_file = os.path.join(
            TEST_DIR, st_dir, "test_llama_inference_ptd.sh")

        self.shell_file_list = [
            llama_pretrain_shell_file,
            llama_inference_shell_file
        ]

    def run_shell(self):
        for shell_file in self.shell_file_list:
            success_check(os.system("sh {}".format(shell_file)))


# ===============================================
# UT test, run with pytest, waiting for more ...
# ===============================================


if __name__ == "__main__":
    test_loader = unittest.TestLoader()
    discover = test_loader.discover(start_dir="../tests/ut", pattern="test*.py")
    runner = unittest.TextTestRunner()
    success_check_ut(runner.run(discover))
    st_test = ST_Test()
    st_test.run_shell()
