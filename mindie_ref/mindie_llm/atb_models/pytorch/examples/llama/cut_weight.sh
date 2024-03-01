# original model path
input_dir="./llama2-7b"
# cutted model path
output_dir="./llama2-7b_parallel"
# cutting parameters
cut_row_keys_=('q_proj','k_proj','v_proj','gate_proj','up_proj')
cut_col_keys_=('o_proj','down_proj')
yi6b=0
is_gqa=0

script_dir=$(cd $(dirname $0); pwd)
transformers_package_path=$(python3 -c 'import transformers; import os; print(os.path.dirname(transformers.__file__))')

function fn_main()
{   
    if [[ ! -z "$1" ]];then
        CUT_OPTION=$1
        echo "[CUT_OPTION]: $CUT_OPTION"
        shift
    fi

    if [[ ! -z "$1" ]];then
        WORLD_SIZE=$1
        echo "[WORLD_SIZE]: $WORLD_SIZE"
        shift
    fi

    until [[ -z "$1" ]]
    do {
        local_arg=$1
        case "${local_arg}" in
        "--yi6b")
            yi6b=1
            ;;
        "--is_gqa")
            is_gqa=1
            ;;
        esac
        shift
    }
    done

    case "${CUT_OPTION}" in
    "--float")
        echo "cutting the float weight..."
        if [[ "${is_gqa}" -eq 1]]; then
            cp $script_dir/../codellama/34b/modeling_llama_cut.py $transformers_package_path/models/llama/modeling_llama.py
        else
            cp $script_dir/modeling_llama_cut.py $transformers_package_path/models/llama/modeling_llama.py
        fi
        python ./cut_float_weight.py \
        --input_path $input_dir \
        --output_path $output_dir \
        --world_size $WORLD_SIZE \
        --cut_row_keys $cut_row_keys_ \
        --cut_col_keys $cut_col_keys_ \
        --is_yi6b $yi6b
        ;;

    "--quant")
        echo "cutting the quant weight..."
        python ./cut_quant_weight.py \
        --input_path $input_dir \
        --output_path $output_dir \
        --world_size $WORLD_SIZE \
        --cut_row_keys $cut_row_keys_ \
        --cut_col_keys $cut_col_keys_
        ;;
    
    "--help")
        echo "cut_weight.sh [--float|--quant] [world_size] [--yi6b] [--is_gqa]"
        ;;

    *)
        echo "unknown build type:${RUN_OPTION}"
        echo "cut_weight.sh [--float|--quant] [world_size] [--yi6b] [--is_gqa]"
        ;;
    esac
}

fn_main "$@"