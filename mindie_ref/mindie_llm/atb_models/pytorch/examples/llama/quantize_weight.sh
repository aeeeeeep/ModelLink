# float model path
input_dir="./llama2-7b"
# ouput quant model path
output_dir="./llama2-7b_quant"
# quant parameters
model_type="Llama"
disable_level="L6"
disable_idx_lst=[0]
quant_task_name="teacher_qualification"

function fn_main()
{
    if [[ ! -z "$1" ]];then
        QUANT_OPTION=$1
        echo "[QUANT_OPTION]: $QUANT_OPTION"
        shift
    fi

    case "${QUANT_OPTION}" in
    "--quant")
        echo "generating the quant weight..."
        python ./quantize_llama_weight.py \
        --input_path $input_dir \
        --output_path $output_dir \
        --disable_level $disable_level \
        ;;

    "--anti_quant")
        echo "generating the anti-outlier quant weight..."
        python ./quantize_llama_antioutlier_weight.py \
        --input_path $input_dir \
        --output_path $output_dir \
        --disable_level $disable_level \
        --disable_idx_lst $disable_idx_lst \
        --quant_task_name $quant_task_name \
        ;;
    
    "--help")
        echo "quantize_weight.sh [--quant|--anti_quant]"
        ;;

    *)
        echo "unknown build type:${RUN_OPTION}"
        echo "quantize_weight.sh [--quant|--anti_quant]"
        ;;
    esac
}

fn_main "$@"