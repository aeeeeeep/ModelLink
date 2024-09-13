import os
from pathlib import Path


def read_files_from_txt(txt_file):
    with open(txt_file, "r") as f:
        return [line.strip() for line in f.readlines()]


def is_examples(file):
    return file.startswith("example/")


def is_pipecase(file):
    return file.startswith("tests/pipeline")


def is_markdown(file):
    return file.endswith(".md")


def skip_ci_file(files, skip_cond):
    for file in files:
        if not any(condition(file) for condition in skip_cond):
            return False
    return True


def alter_skip_ci():
    parent_dir = Path(__file__).absolute().parents[2]
    raw_txt_file = os.path.join(parent_dir, "modify.txt")

    if not os.path.exists(raw_txt_file):
        return False
    
    file_list = read_files_from_txt(raw_txt_file)
    skip_conds = [
        is_examples,
        is_pipecase,
        is_markdown
    ]

    return skip_ci_file(file_list, skip_conds)


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
            test_dir, st_dir, "st_run.sh"
        )

    def run_st(self):
        rectify_case = f"bash {self.st_shell}"
        rectify_code = acquire_exitcode(rectify_case)
        if rectify_code != 0:
            print("rectify case failed, check it.")
            exit(1)


def run_tests():
    ut = UT_Test()
    st = ST_Test()

    ut.run_ut()
    st.run_st()


def main():
    if alter_skip_ci():
        print("Skipping CI")
    else:
        run_tests()

if __name__ == "__main__":
    main()
    