import torch
import os
from transformers import LlamaTokenizer, pipeline, LlamaForCausalLM, AutoTokenizer, AutoModelForCausalLM
# from transformers import LlamaTokenizer, pipeline, AutoTokenizer

# from modeling_llama_parallel import LlamaForCausalLM

# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers.models.llama.configuration_llama import LlamaConfig
# from configuration_llama import LlamaConfig
import argparse

#cut weights
#cut_row_keys :dim 0  cut_col_keys :dim 1  nn.linear: x*A.T
def cut_weights(model,world_size,cut_row_keys=['q_proj','k_proj','v_proj','gate_proj','up_proj'],cut_col_keys=['o_proj','down_proj']):
    # tensor_list=list(model.state_dict().values()) #

    state_dict_list=[{} for i in range(world_size)]
    for key, tensor in model.state_dict().items():
        print(f"key = {key}")
        key_short=key.split('.')[-2]
        print(f"key_short = {key_short}")
        if key_short in cut_row_keys:
            print(f"cut_row_keys, tensor.shape = {tensor.shape}")
            cut_tensor_list = torch.chunk(tensor,world_size,dim=0)
            # if cut_tensor_list[0].shape != cut_tensor_list[1].shape:
            #     print(f"cut_row_keys finish, cut_tensor_list[0].shape = {cut_tensor_list[0].shape}, "
            #         f"cut_tensor_list[1].shape = {cut_tensor_list[1].shape}")

        elif key_short in cut_col_keys:
            print(f"cut_col_keys, tensor.shape = {tensor.shape}")
            cut_tensor_list = torch.chunk(tensor,world_size,dim=1)
            # if cut_tensor_list[0].shape != cut_tensor_list[1].shape:
            #     print(f"cut_col_keys finish, cut_tensor_list[0].shape = {cut_tensor_list[0].shape}, "
            #           f"cut_tensor_list[1].shape = {cut_tensor_list[1].shape}")
        else:
            # print("cut else")
            cut_tensor_list=[tensor]*world_size
        for i in range(world_size):
            state_dict_list[i][key]=cut_tensor_list[i]
    return state_dict_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default = "/data/models/CodeLlama-34b-Instruct-hf/",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default ='/data/models/CodeLlama-34b-Instruct-hf_part/',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default = 2,
        help="world_size",
    )
    parser.add_argument(
        "--cut_row_keys",
        default = ['q_proj','k_proj','v_proj','gate_proj','up_proj'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default = ['o_proj','down_proj'],
        help="cut_col_keys",
    )
    args = parser.parse_args()
    args.world_size=int(args.world_size) 
    tokenizer = AutoTokenizer.from_pretrained(args.input_path) #load the tokrnnizer
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
    tokenizer.save_pretrained(args.output_path+'/tokenizer') #save the tokrnnizer
    
    # model = LlamaForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.float16) #load the raw model
    model = AutoModelForCausalLM.from_pretrained(args.input_path, torch_dtype=torch.bfloat16)  # load the raw model
    state_dict_list=cut_weights(model,args.world_size,args.cut_row_keys,args.cut_col_keys) #cut the weight
    model_config=model.config
    print(f"model_config = {model_config}")
    # create new model config, add the world size parameter, the model size will be cut according to the world size in the model file
    create_config=LlamaConfig(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            hidden_act=model_config.hidden_act,
            max_position_embeddings=model_config.max_position_embeddings,
            initializer_range=model_config.initializer_range,
            rms_norm_eps=model_config.rms_norm_eps,
            rope_scaling=model_config.rope_scaling,
            rope_theta=model_config.rope_theta,
            use_cache=model_config.use_cache,
            pad_token_id=model_config.pad_token_id,
            bos_token_id=model_config.bos_token_id,
            eos_token_id=model_config.eos_token_id,
            tie_word_embeddings=model_config.tie_word_embeddings,
            world_size=args.world_size,
            num_key_value_heads=model_config.num_key_value_heads,
            # max_sequence_length=model_config.max_sequence_length,
            pretraining_tp=model_config.pretraining_tp,
            architectures=model_config.architectures,
            model_type= model_config.model_type,
            torch_dtype= model_config.torch_dtype,
            transformers_version= model_config.transformers_version
    )
    print(f"create_config = {create_config}")
    # create new model according to the model config
    creat_model=LlamaForCausalLM(create_config)
    for i in range(args.world_size):
        print(f"state_dict_list{i} = {state_dict_list[i].keys()}")
        creat_model.load_state_dict(state_dict_list[i]) #load the weights to the model
        creat_model.save_pretrained(args.output_path+'/part_model/'+str(i)+'/') #save model
    print('Tensor parallelism weights have been successfully saved.')


