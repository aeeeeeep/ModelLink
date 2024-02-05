# Copyright Huawei Technologies Co., Ltd. 2023-2024. All rights reserved.
import os
import subprocess
import sys
from pathlib import Path

from setuptools import Extension, setup, find_packages
from setuptools.command.build_ext import build_ext

# Convert distutils Windows platform specifiers to CMake -A arguments
PLAT_TO_CMAKE = {
    "win32": "Win32",
    "win-amd64": "x64",
    "win-arm32": "ARM",
    "win-arm64": "ARM64",
}


class LinuxCMakeExtension(Extension):
    def __init__(self, LinuxExtModulesName: str) -> None:
        super().__init__(name=LinuxExtModulesName, sources=[])
        self.root_path = os.fspath(Path("").resolve())


class LinuxCMakeBuilder(build_ext):
    def build_extension(self, ext: LinuxCMakeExtension) -> None:
        target_path = Path(os.path.join(Path.cwd(), self.get_ext_fullpath(ext.name))).parent.resolve()

        build_type = get_build_type(self.debug)

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={target_path}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={build_type}",
        ]
        build_args = []
        cmake_args += [f"-DEXAMPLE_VERSION_INFO={self.distribution.get_version()}"]

        build_temp = Path(os.path.join(Path(self.build_temp), ext.name))
        if not build_temp.exists():
            build_temp.mkdir(parents=True)

        subprocess.run(
            ["cmake", ext.root_path, *cmake_args], cwd=build_temp, check=True
        )
        subprocess.run(
            ["cmake", "--build", ".", *build_args], cwd=build_temp, check=True
        )


def get_build_type(debug):
    is_debug = int(os.environ.get("DEBUG", 0)) if debug is None else debug
    build_type = "Debug" if is_debug else "Release"
    return build_type


setup(
    name="atb_llm",
    version="0.0.1",
    author="",
    author_email="",
    description="ATB LLM Project",
    long_description="",
    package_dir={'atb_llm': 'atb_llm'},
    package_data={
        '': ['*.xlsx', '*.h5', '*.csv', '*.so', '*.avsc', '*.xml', '*.pkl', '*.sql', '*.ini']
    },
    zip_safe=False,
    python_requires=">=3.7",
)
