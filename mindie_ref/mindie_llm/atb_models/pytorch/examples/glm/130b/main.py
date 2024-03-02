import torch
from generate import add_generation_specific_args, generate
from benchmark import add_performance_specific_args, test_performance
from evaluate import add_evaluation_specific_args, evaluate
from initialize import initialize, initialize_model_and_tokenizer


def add_main_specific_args(parser):
    """Arguments for main function"""
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--generate", 
                       action="store_true", 
                       help="generate texts")
    group.add_argument("--evaluate", 
                       action="store_true", 
                       help="evaluate scores with datasets")
    group.add_argument("--benchmark", 
                       action="store_true", 
                       help="test model performance")
    return parser


def main():
    args = initialize(
        add_generation_specific_args, 
        add_main_specific_args, 
        add_evaluation_specific_args,
        add_performance_specific_args
    )
    model, tokenizer = initialize_model_and_tokenizer(args)

    if args.generate:
        with torch.no_grad():
            generate(model, tokenizer, args)
    elif args.evaluate:
        evaluate(model, tokenizer, args.task, args.data_path)
    elif args.benchmark:
        test_performance(model, args.test_batchs, args.test_seqlen_cases, 
                         args.output_dir)
    else:
        raise RuntimeError("must specify one task from: " \
                           "['generate', 'evaluate', 'benchmark']")


if __name__ == "__main__":
    main()
