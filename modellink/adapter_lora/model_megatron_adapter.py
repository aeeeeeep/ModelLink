from peft import LoraConfig, get_peft_model, PeftModel, LoraModel
import megatron
from megatron.arguments import core_transformer_config_from_args
from megatron.training import get_model
from megatron import get_args
from megatron.core import DistributedDataParallel as DDP
from megatron.model import Float16Module
from megatron.training import unwrap_model

ALL_MODULE_WRAPPER_CLASSNAMES = (DDP, Float16Module, PeftModel, LoraModel)


def get_model_megatron_patch(*args_input):
    def _hook(_module, _x_in, _x_out):
        """ Extract the feature map of model"""
        _x_out.requires_grad_(True)

    def _create_hooks(_model, layer):
        """ Make the hooks function"""
        for name, module in _model.named_modules():
            _name = name.split('.')[-1]
            if _name in layer:
                module.register_forward_hook(_hook)

    model = get_model(*args_input)
    args = get_args()

    if args.lora_target_modules:
        config = core_transformer_config_from_args(args)
        lora_config = LoraConfig(
            r=args.lora_r,
            lora_alpha=args.lora_alpha,
            target_modules=args.lora_target_modules,
            lora_dropout=0.0,
            bias="none",
            megatron_config=config,
            megatron_core="megatron.core",
        )

        for model_item in model:
            model_item = get_peft_model(model_item, lora_config)
            _create_hooks(model_item, args.lora_register_forward_hook)
            model_item.print_trainable_parameters()

    return model


def unwrap_model_megatron_patch(model, module_instances=ALL_MODULE_WRAPPER_CLASSNAMES):
    return unwrap_model(model, module_instances=module_instances)


def apply_model_patch():
    megatron.training.unwrap_model = unwrap_model_megatron_patch
    megatron.checkpointing.unwrap_model = unwrap_model_megatron_patch
    megatron.training.get_model = get_model_megatron_patch
