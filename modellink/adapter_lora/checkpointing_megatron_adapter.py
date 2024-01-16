import os
from functools import wraps
import megatron
from megatron.checkpointing import _load_base_checkpoint, load_checkpoint
from megatron import get_args


def merge_dicts(dict1, dict2):
    result = dict1
    for key, value in dict2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = merge_dicts(result[key], value)
        else:
            result[key] = value
    return result


def modify_keys_with_dict(dictionary, words_to_replace, exclude_words):
    modified_dict = {}
    for key, value in dictionary.items():
        key_str = str(key)
        matched_word = next((word for word, replacement in words_to_replace.items() if word in key_str), None)
        if (matched_word and
                not any(exclude_word in key_str for exclude_word in exclude_words) and
                key_str != matched_word):
            # Check if a word to replace is present in the key and none of the exclude_words are present
            new_key = key_str.replace(matched_word, words_to_replace[matched_word])
            if isinstance(value, dict):
                modified_dict[new_key] = modify_keys_with_dict(value, words_to_replace, exclude_words)
            else:
                modified_dict[new_key] = value
        else:
            if isinstance(value, dict):
                modified_dict[key] = modify_keys_with_dict(value, words_to_replace, exclude_words)
            else:
                modified_dict[key] = value
    return modified_dict


def _load_base_checkpoint_megatron_patch(load_dir, rank0=False):
    args = get_args()
    state_dict, checkpoint_name, release = _load_base_checkpoint(load_dir, rank0)

    if args.lora_target_modules and state_dict is not None:
        words_to_match = {'weight': 'base_layer.weight', 'bias': 'base_layer.bias'}
        exclude_words = ['base_layer', 'lora_', 'norm']
        state_dict = modify_keys_with_dict(state_dict, words_to_match, exclude_words)

        if not args.lora_load:
            return state_dict, checkpoint_name, release

        # Read the tracker file and set the iteration.
        state_dict_lora, checkpoint_name_lora, release_lora = _load_base_checkpoint(args.lora_load, rank0)
        if state_dict_lora is not None:
            merge_dicts(state_dict, state_dict_lora)
            checkpoint_name = checkpoint_name_lora
            release = release_lora
    return state_dict, checkpoint_name, release


def load_checkpoint_megatron_patch(model, optimizer, opt_param_scheduler, load_arg='load', strict=True):
    args = get_args()
    if args.lora_target_modules and (not args.lora_load or not os.path.exists(args.lora_load)):
        strict = False

    return load_checkpoint(model, optimizer, opt_param_scheduler, load_arg=load_arg, strict=strict)


def state_dict_for_save_checkpoint_patch(state_dict):
    state_dict_ = dict()
    for key in state_dict:
        if 'lora' in key:
            state_dict_[key] = state_dict[key]
    return state_dict_


def state_dict_for_save_checkpoint_wrapper(fn):
    @wraps(fn)
    def wrapper(self, prefix='', keep_vars=False):
        args = get_args()
        if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
            return state_dict_for_save_checkpoint_patch(self.state_dict(prefix=prefix, keep_vars=keep_vars))
        return fn(self, prefix='', keep_vars=False)

    return wrapper


def apply_checkpointing_patch():
    megatron.checkpointing._load_base_checkpoint = _load_base_checkpoint_megatron_patch
    megatron.training.load_checkpoint = load_checkpoint_megatron_patch
    megatron.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint \
        = state_dict_for_save_checkpoint_wrapper(
        megatron.model.transformer.ParallelTransformer.state_dict_for_save_checkpoint)
