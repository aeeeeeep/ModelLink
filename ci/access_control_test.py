import os
import stat
import sys
from pathlib import Path
import subprocess


def acquire_exitcode(command):
    exitcode = os.system(command)
    real_code = os.WEXITSTATUS(exitcode)
    return real_code


# =============================
# UT test, run with pytest
# =============================

class UT_Test:

    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')
        self.ut_file = os.path.join(test_dir, "ut")
    
    def run_ut(self):
        command = f"pytest -x {self.ut_file}"
        code = acquire_exitcode(command)
        if code == 0:
            print("UT test success")
        else:
            print("UT failed")
            exit(1)


# ===============================================
# ST test, run with sh.
# ===============================================

class ST_Test:
    
    def __init__(self):

        base_dir = Path(__file__).absolute().parent.parent
        test_dir = os.path.join(base_dir, 'tests')

        st_dir = "st"
        self.st_shell = os.path.join(
            test_dir, st_dir, "run.sh"
        )
        llama_instruction_shell_file = os.path.join(
            test_dir, st_dir, "test_llama_instruction_ptd.sh")
        llama_pretrain_ha_save_shell_file = os.path.join(
            test_dir, st_dir, "test_llama_pretrain_ha_save_ptd.sh")
        llama_pretrain_ha_load_shell_file = os.path.join(
            test_dir, st_dir, "test_llama_pretrain_ha_load_ptd.sh")

        self.st_file_list = [
            llama_instruction_shell_file
        ]

    def run_st(self):
        rectify_case = f"bash {self.st_shell}"
        rectify_code = acquire_exitcode(rectify_case)
        if rectify_code != 0:
            print("rectify case failed, check it.")
            exit(1)
        all_success = True
        for shell_file in self.st_file_list:
            command = f"sh {shell_file}"
            st_exitcode = acquire_exitcode(command)
            if st_exitcode != 0:
                all_success = False
                print(f"ST run {shell_file} failed")
                exit(1)

        if all_success:
            print("ST test success")
        else:
            print("ST failed")
            exit(1)


if __name__ == "__main__":
    ut = UT_Test()
    ut.run_ut()
    # st = ST_Test()
    # st.run_st()