def is_enable_lora():
    from megatron import get_args
    args = get_args()
    if hasattr(args, 'lora_target_modules') and args.lora_target_modules:
        return True
    return False