# 总入口脚本
# 性能测试时，第一次输入一个短句对模型进行warm up
# warm up完成后，输入clear清除history
# 之后开始性能测试，输入测试用例（五条直线相交，最多能有多少个交点？）后等待测试结果即可
# 测试结果将输出到当前目录下的performance.txt文件中

SCRIPT_DIR=$(cd $(dirname $0); pwd)
MODEL_TARGET_DIR=$SCRIPT_DIR
TRANSFORMER_PACKAGE_PATH=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')
RUN_OPTION_LIST="--run --performance --webdemo --zhipu --profiling"
RUN_OPTION="--run"

ACLTRANSFORMER_TESTDATA='/data/acltransformer_testdata'

export ACLTRANSFORMER_CONVERT_NCHW_TO_ND=1
export ACLTRANSFORMER_PLAN_EXECUTE_ASYNC=1
export TASK_QUEUE_ENABLE=1

function fn_prepare()
{
    echo "$RUN_OPTION $SCRIPT_PATH"

    if [ ! -f "$MODEL_TARGET_DIR/pytorch_model-00001-of-00046.bin" ];then
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/merges.txt $MODEL_TARGET_DIR/merges.txt
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model.bin.index.json $MODEL_TARGET_DIR/pytorch_model.bin.index.json
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/special_tokens_map.json $MODEL_TARGET_DIR/special_tokens_map.json
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/tokenizer_config.json $MODEL_TARGET_DIR/tokenizer_config.json
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/tokenizer.json $MODEL_TARGET_DIR/tokenizer.json
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/vocab.json $MODEL_TARGET_DIR/vocab.json
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00001-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00001-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00002-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00002-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00003-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00003-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00004-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00004-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00005-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00005-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00006-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00006-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00007-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00007-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00008-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00008-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00009-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00009-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00010-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00010-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00011-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00011-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00012-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00012-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00013-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00013-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00014-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00014-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00015-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00015-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00016-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00016-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00017-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00017-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00018-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00018-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00019-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00019-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00020-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00020-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00021-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00021-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00022-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00022-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00023-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00023-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00024-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00024-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00025-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00025-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00026-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00026-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00027-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00027-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00028-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00028-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00029-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00029-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00030-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00030-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00031-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00031-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00032-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00032-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00033-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00033-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00034-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00034-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00035-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00035-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00036-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00036-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00037-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00037-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00038-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00038-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00039-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00039-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00040-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00040-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00041-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00041-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00042-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00042-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00043-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00043-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00044-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00044-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00045-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00045-of-00046.bin
        ln -s $ACLTRANSFORMER_TESTDATA/weights/gptneox20b/pytorch_model-00046-of-00046.bin $MODEL_TARGET_DIR/pytorch_model-00046-of-00046.bin
    fi
}

function fn_clean()
{
    rm $MODEL_TARGET_DIR/pytorch_model*
    rm $MODEL_TARGET_DIR/merges.txt
    rm $MODEL_TARGET_DIR/pytorch_model.bin.index.json
    rm $MODEL_TARGET_DIR/special_tokens_map.json
    rm $MODEL_TARGET_DIR/tokenizer_config.json
    rm $MODEL_TARGET_DIR/tokenizer.json
    rm $MODEL_TARGET_DIR/vocab.json
}

function fn_main()
{
    echo "-----run.sh-----"
    fn_clean

    if [[ ! -z "$1" ]];then
        RUN_OPTION=$1
    fi
    cd $SCRIPT_DIR

    fn_prepare

    case "${RUN_OPTION}" in
        "--run")
            python3 $SCRIPT_DIR/run_gptneox.py
            ;;
        "--performance")
            ;;
        "--webdemo")
            ;;
        "--zhipu")
            python3 $SCRIPT_DIR/zhipu_test.py
            ;;
        "--profiling")
            ;;
        "--precision")
            python3 $SCRIPT_DIR/run_gptneox.py --precision True
            ;;
        "--help")
            echo "run.sh [model script path] [--run|--performance|--webdemo|--zhipu|--profiling]"
            ;;
        *)
            echo "unknown build type:${RUN_OPTION}"
            echo "run.sh [model script path] [--run|--performance|--webdemo|--zhipu|--profiling]"
            ;;
    esac

    fn_clean
}

fn_main "$@"
