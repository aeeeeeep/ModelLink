# coding=utf-8
# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Setup for pip package."""

import sys
import setuptools

if sys.version_info < (3,):
    raise Exception("Python 2 is not supported by AscendSpeed.")

with open("README.md", "r") as fh:
    long_description = fh.read()


###############################################################################
#                             Dependency Loading                              #
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% #


def req_file(filename):
    with open(filename) as f:
        content = f.readlines()
    return [x.strip() for x in content]


install_requires = req_file("requirements.txt")
cmd_class = {}
exts = []
try:
    import imp
    import os
    from torch_npu.utils import cpp_extension
    from torch.utils.cpp_extension import BuildExtension

    ATB_HOME_PATH = os.environ.get("ATB_HOME_PATH")
    ASCEND_TOOLKIT_HOME = os.environ.get("ASCEND_TOOLKIT_HOME")
    ext_ops = cpp_extension.NpuExtension(
        name="ascendspeed_te_ops",
        sources=["ascendspeed/te/ops/csrc/pybind.cpp",
                 "ascendspeed/te/ops/csrc/flash_attention.cpp",
                 "ascendspeed/te/ops/csrc/add_demo.cpp"],
        extra_compile_args=[
        '-Wno-sign-compare',
        '-Wno-deprecated-declarations',
        '-Wno-return-type',
        '-D__FILENAME__=\"$(notdir $(abspath $<))\"',
        '-I' + os.path.join(imp.find_module('torch_npu')[1], "include", "third_party", "acl", "inc"),
        '-I' + os.path.join(ATB_HOME_PATH, "include"),
        '-I' + os.path.join(ASCEND_TOOLKIT_HOME, "include"),
    ],
        extra_link_args=[
        '-L' + os.path.join(ATB_HOME_PATH, "lib"),
        '-latb_torch'
    ],
    )
    exts.append(ext_ops)
    cmd_class = {"build_ext": BuildExtension.with_options(use_ninja=False)}
except Exception:
    print('Can not find any torch_npu, ops setup failed')

setuptools.setup(
    package_data={'ascendspeed':['ascendspeed/data/Makefile']},
    name="ascendspeed",
    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version="0.0.1",
    description="AscendSpeed for LLMs of Ascend",
    long_description="AscendSpeed for LLMs of Ascend",
    long_description_content_type="text/markdown",
    # The project's main homepage.
    url="https://gitee.com/ascend/AscendSpeed",
    author="Ascend",
    maintainer="Ascend",
    # The licence under which the project is released
    license="See https://gitee.com/ascend/AscendSpeed",
    classifiers=[
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Information Technology',
        # Indicate what your project relates to
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
        # Supported python versions
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        # Additional Setting
        'Environment :: Console',
        'Natural Language :: English',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.6',
    packages=setuptools.find_packages(),
    install_requires=install_requires,
    # Add in any packaged data.
    include_package_data=True,
    zip_safe=False,
    # PyPI package information.
    keywords="Ascend, langauge, deep learning, NLP",
    cmdclass=cmd_class,
    ext_modules=exts
)
