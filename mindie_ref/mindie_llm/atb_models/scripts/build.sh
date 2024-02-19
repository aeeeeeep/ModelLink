#!/bin/bash
# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


set -ex
SCRIPT_DIR=$(cd $(dirname -- $0); pwd)
CURRENT_DIR=$(pwd)
cd $SCRIPT_DIR
TARGET_FRAMEWORK=torch
USE_CXX11_ABI=$(python3 get_cxx11_abi_flag.py -f "${TARGET_FRAMEWORK}")
ARCH="aarch64"
if [ $( uname -a | grep -c -i "x86_64" ) -ne 0 ]; then
    ARCH="x86_64"
    echo "it is system of x86_64"
elif [ $( uname -a | grep -c -i "aarch64" ) -ne 0 ]; then
    echo "it is system of aarch64"
else
    echo "it is not system of aarch64 or x86_64"
fi
cd ..
export CODE_ROOT=`pwd`
export CACHE_DIR=$CODE_ROOT/build
export OUTPUT_DIR=$CODE_ROOT/output
THIRD_PARTY_DIR=$CODE_ROOT/3rdparty
PACKAGE_NAME="7.0.T800"
ASCEND_SPEED_VERSION=""
VERSION_B=""
README_DIR=$CODE_ROOT
COMPILE_OPTIONS=""
INCREMENTAL_SWITCH=OFF
HOST_CODE_PACK_SWITCH=ON
DEVICE_CODE_PACK_SWITCH=ON
USE_VERBOSE=OFF
IS_RELEASE=0
BUILD_OPTION_LIST="3rdparty download_testdata unittest unittest_and_run pythontest pythontest_and_run debug release help python_unittest_and_run master"
BUILD_CONFIGURE_LIST=("--output=.*" "--cache=.*" "--verbose" "--incremental" "--gcov" "--no_hostbin" "--no_devicebin" "--use_cxx11_abi=0"
    "--use_cxx11_abi=1" "--build_config=.*" "--optimize_off" "--use_torch_runner" "--use_lccl_runner" "--use_hccl_runner" "--doxygen" "--no_warn" 
    "--ascend_speed_version=.*" "--release_b_version=.*")

function export_speed_env()
{
    cd $OUTPUT_DIR/atb_speed
    source set_env.sh
}

function fn_build_googletest()
{
    if [ -d "$THIRD_PARTY_DIR/googletest/lib" -a -d "$THIRD_PARTY_DIR/googletest/include" ]; then
        return $?
    fi
    cd $CACHE_DIR
    wget --no-check-certificate https://github.com/google/googletest/archive/refs/tags/v1.13.0.tar.gz
    tar -xf v1.13.0.tar.gz
    cd googletest-1.13.0
    mkdir build
    cd build
    if [ "$USE_CXX11_ABI" == "ON" ]
    then
        sed -i '4 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=1)' ../CMakeLists.txt
    else
        sed -i '4 a add_compile_definitions(_GLIBCXX_USE_CXX11_ABI=0)' ../CMakeLists.txt
    fi
    cmake .. -DCMAKE_INSTALL_PREFIX=$THIRD_PARTY_DIR/googletest -DCMAKE_SKIP_RPATH=TRUE -DCMAKE_CXX_FLAGS="-fPIC"
    cmake --build . --parallel $(nproc)
    cmake --install .
    [[ -d "$THIRD_PARTY_DIR/googletest/lib64" ]] && cp -rf $THIRD_PARTY_DIR/googletest/lib64 $THIRD_PARTY_DIR/googletest/lib
    echo "Googletest is successfully installed to $THIRD_PARTY_DIR/googletest"
}

function fn_run_unittest()
{
    export_speed_env
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    export LD_LIBRARY_PATH=$PYTORCH_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    if [ -z "${PYTORCH_NPU_INSTALL_PATH}" ];then
        export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
    fi
    export LD_LIBRARY_PATH=$PYTORCH_NPU_INSTALL_PATH/lib:$LD_LIBRARY_PATH
    echo "run $OUTPUT_DIR/atb_speed/bin/speed_unittest"
    $OUTPUT_DIR/atb_speed/bin/speed_unittest --gtest_filter=-*.TestAllGatherHccl:*.TestBroadcastHccl --gtest_output=xml:test_detail.xml
}

function fn_build_stub()
{
    if [[ -f "$THIRD_PARTY_DIR/googletest/include/gtest/stub.h" ]]; then
        return $?
    fi
    cd $CACHE_DIR
    rm -rf cpp-stub-master.tar.gz
    wget --no-check-certificate https://github.com/coolxv/cpp-stub/archive/refs/heads/master.tar.gz
    tar -zxvf master.tar.gz
    cp $CACHE_DIR/cpp-stub-master/src/stub.h $THIRD_PARTY_DIR/googletest/include/gtest
    rm -rf $CACHE_DIR/cpp-stub-master/
}

function fn_build_3rdparty_for_test()
{
    if [ -d "$CACHE_DIR" ]
    then
        rm -rf $CACHE_DIR
    fi
    mkdir $CACHE_DIR
    cd $CACHE_DIR
    fn_build_googletest
    fn_build_stub
    cd ..
}

function fn_build_nlohmann_json()
{
    NLOHMANN_DIR=$THIRD_PARTY_DIR/nlohmannJson/include
    if [ ! -d "$NLOHMANN_DIR" ];then
        if [ ! -f "$CACHE_DIR/nlohmann/include.zip" ];then
            cd $CACHE_DIR
            rm -rf nlohmann
            mkdir nlohmann
            cd nlohmann
            wget --no-check-certificate https://ascend-cann.obs.cn-north-4.myhuaweicloud.com/include.zip
        else
            cd $CACHE_DIR/nlohmann
        fi
        unzip include.zip
        mkdir -p $THIRD_PARTY_DIR/nlohmannJson
        cp -r ./include $THIRD_PARTY_DIR/nlohmannJson
        cd $CACHE_DIR
        rm -rf nlohmann
    fi
}

function fn_build_3rdparty()
{
    rm -rf $CACHE_DIR
    mkdir $CACHE_DIR
    cd $CACHE_DIR
    fn_build_nlohmann_json
    cd ..
}

function fn_init_pytorch_env()
{
    export PYTHON_INCLUDE_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTHON_LIB_PATH="$(python3 -c 'from sysconfig import get_paths; print(get_paths()["include"])')"
    export PYTORCH_INSTALL_PATH="$(python3 -c 'import torch, os; print(os.path.dirname(os.path.abspath(torch.__file__)))')"
    if [ -z "${PYTORCH_NPU_INSTALL_PATH}" ];then
        export PYTORCH_NPU_INSTALL_PATH="$(python3 -c 'import torch, torch_npu, os; print(os.path.dirname(os.path.abspath(torch_npu.__file__)))')"
    fi
    echo "PYTHON_INCLUDE_PATH=$PYTHON_INCLUDE_PATH"
    echo "PYTHON_LIB_PATH=$PYTHON_LIB_PATH"
    echo "PYTORCH_INSTALL_PATH=$PYTORCH_INSTALL_PATH"
    echo "PYTORCH_NPU_INSTALL_PATH=$PYTORCH_NPU_INSTALL_PATH"

    COUNT=$(grep get_tensor_npu_format ${PYTORCH_NPU_INSTALL_PATH}/include/torch_npu/csrc/framework/utils/CalcuOpUtil.h | wc -l)
    if [ "$COUNT" == "1" ];then
        echo "use get_tensor_npu_format"
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_GET_TENSOR_NPU_FORMAT_OLD=ON"
    else
        echo "use GetTensorNpuFormat"
    fi

    COUNT=$(grep SetCustomHandler ${PYTORCH_NPU_INSTALL_PATH}/include/torch_npu/csrc/framework/OpCommand.h | wc -l)
    if [ $COUNT -ge 1 ];then
        echo "use SetCustomHandler"
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_SETCUSTOMHANDLER=ON"
    else
        echo "not use SetCustomHandler"
    fi

    is_higher_PTA6=$(nm --dynamic ${PYTORCH_NPU_INSTALL_PATH}/lib/libtorch_npu.so | grep _ZN6at_npu6native17empty_with_formatEN3c108ArrayRefIlEERKNS1_13TensorOptionsElb | wc -l)
    if [ $is_higher_PTA6 -ge 1 ];then
        echo "using pta verion after PTA6RC1B010 (6.0.RC1.B010)"
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_HIGHER_THAN_PTA6=ON"
    else
        echo "using pta version below PTA6RC1B010 (6.0.RC1.B010)"
    fi
}

function fn_run_pythontest()
{
    cd $OUTPUT_DIR/atb_speed
    source set_env.sh
    cd $CODE_ROOT/tests/layertest/
    rm -rf ./kernel_meta*
    export ATB_CONVERT_NCHW_TO_ND=1
    export HCCL_WHITELIST_DISABLE=1
    python3 -m unittest discover -s . -p "*.py"
}

function fn_build_coverage()
{
    GCOV_DIR=$OUTPUT_DIR/atb_speed/gcov
    GCOV_CACHE_DIR=$OUTPUT_DIR/atb_speed/gcov/cache
    GCOV_INFO_DIR=$OUTPUT_DIR/atb_speed/gcov/cov_info
    LCOV_PATH=`which lcov`
    GENHTML_PATH=`which genhtml`
    FIND_IGNORE_PATH=$CACHE_DIR/core/CMakeFiles/atb_speed_static.dir/*
    if [ -d "$GCOV_DIR" ]
    then
        rm -rf $GCOV_DIR
    fi
    mkdir $GCOV_DIR
    mkdir $GCOV_CACHE_DIR
    mkdir $GCOV_INFO_DIR

    $LCOV_PATH -d $GCOV_CACHE_DIR --zerocounters >> $GCOV_DIR/log.txt

    find $CACHE_DIR -not -path "$FIND_IGNORE_PATH" -name "*.gcno" | xargs -i cp {} $GCOV_CACHE_DIR
    $LCOV_PATH -c -i -d $GCOV_CACHE_DIR -o $GCOV_INFO_DIR/init.info >> $GCOV_DIR/log.txt
    
    [[ "$COVERAGE_TYPE" == "unittest" ]] && fn_run_unittest
    [[ "$COVERAGE_TYPE" == "pythontest" ]] && fn_run_pythontest

    find $CACHE_DIR -name "*.gcda" | xargs -i cp {} $GCOV_CACHE_DIR
    cd $GCOV_CACHE_DIR
    find . -name "*.cpp" | xargs -i gcov {} >> $GCOV_DIR/log.txt
    cd ..
    $LCOV_PATH -c -d $GCOV_CACHE_DIR -o $GCOV_INFO_DIR/cover.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $LCOV_PATH -a $GCOV_INFO_DIR/init.info -a $GCOV_INFO_DIR/cover.info -o $GCOV_INFO_DIR/total.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $LCOV_PATH --remove $GCOV_INFO_DIR/total.info '*/3rdparty/*' '*torch/*' '*c10/*' '*ATen/*' '*/c++/7*' '*tests/*' '*tools/*' '/usr/*' '/opt/*' '*models/*' -o $GCOV_INFO_DIR/final.info --rc lcov_branch_coverage=1 >> $GCOV_DIR/log.txt
    $GENHTML_PATH --rc lcov_branch_coverage=1 -o cover_result $GCOV_INFO_DIR/final.info -o cover_result >> $GCOV_DIR/log.txt
    tail -n 4 $GCOV_DIR/log.txt
    cd $OUTPUT_DIR/atb_speed
    tar -czf gcov.tar.gz gcov
    rm -rf gcov
}

function fn_build_version_info()
{
    if [ -f "$CODE_ROOT"/../../../../CI/config/version.ini ]; then
        PACKAGE_NAME=$(cat $CODE_ROOT/../../../../CI/config/version.ini | grep "PackageName" | cut -d "=" -f 2)
        VERSION=$(cat "$CODE_ROOT"/../../../../CI/config/version.ini | grep "ATBVersion" | cut -d "=" -f 2)
        ASCEND_SPEED_VERSION=$(cat $CODE_ROOT/../../../../CI/config/version.ini | grep "ATB-ModelsVersion" | cut -d "=" -f 2)
    fi
    current_time=$(date +"%Y-%m-%d %r %Z")
    touch $OUTPUT_DIR/atb_speed/version.info
    cat > $OUTPUT_DIR/atb_speed/version.info <<EOF
ATBVersion : ${VERSION_B}
ModelsVersion : ${ASCEND_SPEED_VERSION}
Platform : ${ARCH}
Time: ${current_time}
EOF

}

function fn_build_for_ci()
{
    cd $OUTPUT_DIR/atb_speed
    rm -rf ./*.tar.gz
    cp $CODE_ROOT/dist/atb_llm*.whl .
    cp -r $CODE_ROOT/atb_llm .
    cp $CODE_ROOT/setup.py .
    cp -r $CODE_ROOT/examples .
    cp -r $CODE_ROOT/tests .
    cp $README_DIR/README.md .
    fn_build_version_info

    torch_vision=$(pip list | grep torch | head  -n 1 | awk '{print $2}' | cut -d '+' -f1)
    if [ "$USE_CXX11_ABI" == "OFF" ];then
        abi=0
    else
        abi=1
    fi

    tar_package_name="Ascend-mindie-atb-models_${PACKAGE_NAME}_linux-${ARCH}_torch${torch_vision}-abi${abi}.tar.gz"

    if [ $IS_RELEASE -eq 1 ]; then
        source_folder_list=$(cat $SCRIPT_DIR/release_folder.ini | xargs)
        tar czf $tar_package_name $source_folder_list --owner=0 --group=0
    else
        tar czf $tar_package_name ./* --owner=0 --group=0
    fi

    if [ -f "README.md" ];then
        rm -rf README.md
    fi
}

function fn_make_whl() {
    echo "make atb_llm whl package"
    cd $CODE_ROOT
    python3 $CODE_ROOT/setup.py bdist_wheel
}

function fn_build()
{
    fn_build_3rdparty
    if [ ! -d "$OUTPUT_DIR" ];then
        mkdir -p $OUTPUT_DIR
    fi
    if [ "$INCREMENTAL_SWITCH" == "OFF" ];then
        rm -rf $CACHE_DIR
    fi
    if [ ! -d "$CACHE_DIR" ];then
        mkdir $CACHE_DIR
    fi
    cd $CACHE_DIR
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_INSTALL_PREFIX=$OUTPUT_DIR/atb_speed"

    cxx11_flag_str="--use_cxx11_abi"
    if [[ "$COMPILE_OPTIONS" == *$cxx11_flag_str* ]]
    then
    echo "compile_options contain cxx11_abi"
    else
    COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=${USE_CXX11_ABI}"
    fi

    echo "COMPILE_OPTIONS:$COMPILE_OPTIONS"
    cmake $CODE_ROOT $COMPILE_OPTIONS
    if [ "$INCREMENTAL_SWITCH" == "OFF" ];then
        make clean
    fi
    if [ "$USE_VERBOSE" == "ON" ];then
        VERBOSE=1 make -j
    else
        make -j
    fi
    make install
    fn_make_whl
    fn_build_for_ci
}

function fn_main()
{
    if [ -z $ATB_HOME_PATH ];then
        echo "env ATB_HOME_PATH not exist, please source atb's set_env.sh"
        exit -1
    fi

    PYTORCH_VERSION="$(python3 -c 'import torch; print(torch.__version__)')"
    if [ ${PYTORCH_VERSION:0:5} == "1.8.0" ] || [ ${PYTORCH_VERSION:0:4} == "1.11" ];then
        COMPILE_OPTIONS="${COMPILE_OPTIONS} -DTORCH_18=ON"
    fi

    if [[ "$BUILD_OPTION_LIST" =~ "$1" ]];then
        if [[ -z "$1" ]];then
            arg1="master"
        else
            arg1=$1
            shift
        fi
    else
        cfg_flag=0
        for item in ${BUILD_CONFIGURE_LIST[*]};do
            if [[ $1 =~ $item ]];then
                cfg_flag=1
                break 1
            fi
        done
        if [[ $cfg_flag == 1 ]];then
            arg1="master"
        else
            echo "argument $1 is unknown, please type build.sh help for more imformation"
            exit -1
        fi
    fi

    until [[ -z "$1" ]]
    do {
        arg2=$1
        case "${arg2}" in
        --ascend_speed_version=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the ascend_speed_version is not set. This should be set like --ascend_speed_version=<version>"
            else
                ASCEND_SPEED_VERSION=$arg2
            fi
            ;;
        --release_b_version=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the release_b_version is not set. This should be set like --release_b_version=<version>"
            else
                VERSION_B=$arg2
            fi
            ;;
        --output=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the output directory is not set. This should be set like --output=<outputDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export OUTPUT_DIR=$(cd $arg2; pwd)
            fi
            ;;
        --cache=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the cache directory is not set. This should be set like --cache=<cacheDir>"
            else
                cd $CURRENT_DIR
                if [ ! -d "$arg2" ];then
                    mkdir -p $arg2
                fi
                export CACHE_DIR=$(cd $arg2; pwd)
            fi
            ;;
        "--use_cxx11_abi=1")
            USE_CXX11_ABI=ON
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=ON"
            ;;
        "--use_cxx11_abi=0")
            USE_CXX11_ABI=OFF
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_CXX11_ABI=OFF"
            ;;
        "--no_warn")
            ENABLE_WARNINGS=OFF
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DENABLE_WARNINGS=OFF"
            ;;
        "--verbose")
            USE_VERBOSE=ON
            ;;
        "--incremental")
            INCREMENTAL_SWITCH=ON
            ;;
        "--optimize_off")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_OPTIMIZE=OFF"
            ;;
        --link_python=*)
            arg2=${arg2#*=}
            if [ -z $arg2 ];then
                echo "the python version is not set. This should be set like --link_python=python3.7|python3.8|python3.9"
            else
                COMPILE_OPTIONS="${COMPILE_OPTIONS} -DLINK_PYTHON=$arg2"
            fi
            ;;
        "--use_torch_runner")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_TORCH_RUNNER=ON"
            ;;
        esac
        shift
    }
    done

    fn_init_pytorch_env
    case "${arg1}" in
        "download_testdata")
            fn_download_testdata
            ;;
        "debug")
            COMPILE_OPTIONS="${COMPILE_OPTIONS}"
            fn_build
            ;;
        "pythontest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_PYTHON_TEST=ON"
            export COVERAGE_TYPE="pythontest"
            fn_build
            fn_build_coverage
            ;;
        "unittest")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DUSE_UNIT_TEST=ON"
            export COVERAGE_TYPE="unittest"
            fn_build_3rdparty_for_test
            fn_build
            fn_run_unittest
            ;;
        "master")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            fn_build
            ;;
        "release")
            COMPILE_OPTIONS="${COMPILE_OPTIONS} -DCMAKE_BUILD_TYPE=Release"
            IS_RELEASE=1
            fn_build
            ;;
        "help")
            echo "build.sh 3rdparty|unittest|unittest_and_run|pythontest|pythontest_and_run|debug|release|master --incremental|--gcov|--no_hostbin|--no_devicebin|--output=<dir>|--cache=<dir>|--use_cxx11_abi=0|--use_cxx11_abi=1|--build_config=<path>"
            ;;
        *)
            echo "unknown build type:${arg1}";
            ;;
    esac
}

fn_main "$@"