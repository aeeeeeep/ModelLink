import torch
import os
from transformers import AutoTokenizer, AutoModelForCausalLM, MixtralForCausalLM
from transformers.models.mixtral.configuration_mixtral import MixtralConfig
import argparse

#cut weights
#cut_row_keys :dim 0  cut_col_keys :dim 1  nn.linear: x*A.T
def cut_weights(model,world_size,cut_row_keys=['q_proj','k_proj','v_proj', 'w1', 'w3'],cut_col_keys=['o_proj', 'w2']):
    # tensor_list=list(model.state_dict().values()) #
    print("===============  Start cutting weights  ====================")
    print("world_size is: ", world_size)

    state_dict_list=[{} for i in range(world_size)]
    for key, tensor in model.state_dict().items():
        print("key is: ", key)
        key_short=key.split('.')[-2]
        print("key_short is: ", key_short)
        cut_tensor_list_t = []
        if key_short in cut_row_keys:
            print("cut type: row cut")
            cut_tensor_list = torch.chunk(tensor,world_size,dim=0)
        elif key_short in cut_col_keys:
            print("cut type: col cut")
            cut_tensor_list = torch.chunk(tensor,world_size,dim=1)
        else:
            print("cut type: copy")
            cut_tensor_list=[tensor]*world_size
        for tensor in cut_tensor_list:
            cut_tensor_list_t.append(tensor.clone())
        for i in range(world_size):
            state_dict_list[i][key]=cut_tensor_list_t[i]
    print("===============  Cut weights success  ====================")
    return state_dict_list

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Cut Model weights.")
    parser.add_argument(
        "--input_path",
        default = "/home/data/acltransformer_testdata/weights/Mixtral-8x7B-v0.1",
        help="Location of Model weights, which contains model folders",
    )
    parser.add_argument(
        "--output_path",
        default ='/data/models/llama-13b-part_model_2',
        help="Location to write the part weights",
    )
    parser.add_argument(
        "--world_size",
        default = 8,
        help="world_size",
    )
    parser.add_argument(
        "--cut_row_keys",
        default = ['q_proj','k_proj','v_proj', 'w1', 'w3'],
        help="cut_row_keys",
    )
    parser.add_argument(
        "--cut_col_keys",
        default = ['o_proj', 'w2'],
        help="cut_col_keys",
    )
    args = parser.parse_args()
    args.world_size=int(args.world_size) 

    tokenizer_path = args.input_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=False, padding_side='left')
    # tokenizer.add_special_tokens({'pad_token': '[PAD]'})  # pad or not
    tokenizer.save_pretrained(args.output_path+'/tokenizer') #save the tokrnnizer
    
    part_model_path=args.input_path
    model = AutoModelForCausalLM.from_pretrained(part_model_path, torch_dtype=torch.float16)
    state_dict_list=cut_weights(model,args.world_size,args.cut_row_keys) #cut the weight
    model_config=model.config
    print("===============  Config create success  ====================")
    # create new model config, add the world size parameter, the model size will be cut according to the world size in the model file
    create_config=MixtralConfig(
            vocab_size=model_config.vocab_size,
            hidden_size=model_config.hidden_size,
            intermediate_size=model_config.intermediate_size,
            num_hidden_layers=model_config.num_hidden_layers,
            num_attention_heads=model_config.num_attention_heads,
            hidden_act=model_config.hidden_act,
            max_position_embeddings=model_config.max_position_embeddings,
            initializer_range=model_config.initializer_range,
            rms_norm_eps=model_config.rms_norm_eps,
            use_cache=model_config.use_cache,
            pad_token_id=model_config.pad_token_id,
            bos_token_id=model_config.bos_token_id,
            eos_token_id=model_config.eos_token_id,
            tie_word_embeddings=model_config.tie_word_embeddings,
            world_size=args.world_size,
            # max_sequence_length=model_config.max_sequence_length,
            architectures=model_config.architectures,
            model_type= model_config.model_type,
            torch_dtype= model_config.torch_dtype,
            transformers_version= model_config.transformers_version,
            attention_dropout = model_config.attention_dropout,
            num_experts_per_tok = model_config.num_experts_per_tok,
            num_key_value_heads = model_config.num_key_value_heads,
            num_local_experts = model_config.num_local_experts,
            output_router_logits = model_config.output_router_logits,
            rope_theta = model_config.rope_theta,
            router_aux_loss_coef = model_config.router_aux_loss_coef,
            sliding_window = model_config.sliding_window,
    )
    # create new model according to the model config
    print("===============  Creating model  ====================")
    creat_model=MixtralForCausalLM(create_config)
    print("===============  Model create success, saving models, estimated 10 minutes  ====================")
    for i in range(args.world_size):
        creat_model.load_state_dict(state_dict_list[i]) #load the weights to the model
        creat_model = creat_model.half()
        creat_model.save_pretrained(args.output_path+'/part_model/'+str(i)+'/') #save model
    print('Tensor parallelism weights have been successfully saved.')

