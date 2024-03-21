from functools import partial
import os
import re
import stat
from typing import List, Tuple

import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu

from SwissArmyTransformer import mpu
from SwissArmyTransformer.generation.utils import timed_name, generate_continually
from evaluation.model import batch_filling_sequence
from generation import BeamSearchStrategy, BaseStrategy
from initialize import initialize, initialize_model_and_tokenizer


def add_generation_specific_args(parser):
    parser.add_argument(
        "--sampling-strategy", type=str,
        default="BaseStrategy", 
        help="Type of sampling strategy."
    )
    parser.add_argument(
        "--min-gen-length", type=int, 
        default=0,
        help="The minimum length each blank should generate."
    )
    parser.add_argument(
        "--print-all-beams", 
        action="store_true", 
        help="Print all output generated by beam search strategy."
    )


def isEnglish(s):
    try:
        s.encode(encoding="utf-8").decode("ascii")
    except UnicodeDecodeError:
        return False
    else:
        return True


def get_masks_and_position_ids(seq, mask_position, max_gen_length, gmask=False):
    context_length = seq.shape[1]
    tokens = torch.nn.functional.pad(
        seq, (0, max_gen_length), mode="constant", value=-1)
    attention_mask = torch.ones(
        (1, tokens.shape[-1], tokens.shape[-1]), device=tokens.device)
    attention_mask.tril_()
    attention_mask[..., : context_length - 1] = 1
    attention_mask.unsqueeze_(1)
    attention_mask = (attention_mask < 0.5).bool()

    position_ids = torch.arange(
        tokens.shape[-1], dtype=torch.long, device=tokens.device)
    if not gmask:
        position_ids[context_length - 1:] = mask_position

    position_ids = position_ids.unsqueeze(0)

    return tokens, attention_mask, position_ids


def fill_blanks(raw_text: str, model, tokenizer, strategy, args) -> Tuple[List[str], List[str], List[List[str]]]:
    # add MASK
    generation_mask = "[gMASK]"
    if "[MASK]" in raw_text:
        generation_mask = "[MASK]"
    elif "[sMASK]" in raw_text:
        generation_mask = "[sMASK]"
    use_gmask = "[MASK]" not in raw_text and "[sMASK]" not in raw_text

    mask_pattern = r"\[[sg]?MASK\]"
    text_list = re.split(mask_pattern, raw_text)
    pattern_list = re.compile(mask_pattern).findall(raw_text)
    seq = []
    for i, pattern in enumerate(pattern_list):
        sub_text = text_list[i]
        seq.extend(tokenizer.tokenize(sub_text))
        seq.append(tokenizer.get_command(pattern))

    seq.extend(tokenizer.tokenize(text_list[-1]))

    if "MASK]" not in raw_text:
        seq += [tokenizer.get_command(generation_mask)]
        raw_text += " " + generation_mask
    if not raw_text.endswith("MASK]"):
        seq = seq + [tokenizer.get_command("eos")]
    if mpu.get_model_parallel_rank() == 0:
        print("\nInput: {}\n".format(raw_text))
    if len(seq) > args.max_sequence_length:
        raise ValueError("text too long.")

    # generation
    is_english = isEnglish(raw_text)
    output_list = [seq]
    num_output = args.num_beams if args.sampling_strategy == "BeamSearchStrategy" else 1
    last_pos, answers, answers_with_style, blanks = (
        [0] * num_output,
        ["" for _ in range(num_output)],
        ["" for _ in range(num_output)],
        [[] for _ in range(num_output)],
    )

    # continually detect the first mark position
    while True:
        seq = output_list[0]
        # detect mask position
        mask_token = tokenizer.get_command(generation_mask)
        if mask_token not in seq:
            break
        mask_position = seq.index(mask_token)

        output_list = []

        input_seq = torch.cuda.LongTensor(
            [seq + [tokenizer.get_command("sop")]],
            device=args.device,
        )
        output, _ = batch_filling_sequence(
            model,
            input_seq,
            torch.cuda.LongTensor([input_seq.shape[-1]], device=args.device),
            strategy=strategy,
            get_masks_and_position_ids=partial(
                get_masks_and_position_ids,
                mask_position=mask_position,
                max_gen_length=args.out_seq_length - input_seq.shape[-1],
                gmask=use_gmask,
            ),
        )
        if isinstance(output, torch.Tensor):  # different strategies
            output = output.tolist()
        output = output[0]  # batch_size = 1
        output_list.extend(output)

        # clip -1s and fill back generated things into seq
        for i, output in enumerate(output_list):
            output = output.tolist() if isinstance(
                output, torch.Tensor) else output
            try:
                unfinished = output.index(-1)
            except ValueError:
                unfinished = len(output)
            if output[unfinished - 1] in strategy.end_tokens:
                unfinished -= 1
            bog = output.index(tokenizer.get_command("sop"))

            prefix = tokenizer.detokenize(output[last_pos[i]: mask_position])
            blank = tokenizer.detokenize(output[bog + 1: unfinished])
            answers_with_style[i] += (
                prefix
                + (" " if is_english else "")
                + ("\033[4m" if use_gmask else "\x1b[0;32m\033[4m")
                + blank
                + ("\033[0m" if use_gmask else "\033[0m\x1b[0m")
                + (" " if is_english else "")
            )
            blanks[i].append(blank)
            last_pos[i] = mask_position + unfinished - (bog + 1)
            output_list[i] = output[:mask_position] + output[bog +
                                                             1: unfinished] + output[mask_position + 1: bog]

    for i, output in enumerate(output_list):
        if output[-1] == tokenizer.get_command("eos"):
            output = output[:-1]
        answers_with_style[i] += tokenizer.detokenize(output[last_pos[i]:])
        answers[i] = tokenizer.detokenize(output)

    return answers, answers_with_style, blanks


def generate(model, tokenizer, args):
    end_tokens = [tokenizer.get_command("eop"), tokenizer.get_command("eos")]

    if args.sampling_strategy == "BaseStrategy":
        strategy = BaseStrategy(
            batch_size=1, temperature=args.temperature, top_k=args.top_k, top_p=args.top_p, end_tokens=end_tokens
        )
    elif args.sampling_strategy == "BeamSearchStrategy":
        strategy = BeamSearchStrategy(
            1,
            args.num_beams,
            length_penalty=args.length_penalty,
            consider_end=True,
            end_tokens=end_tokens,
            no_repeat_ngram_size=args.no_repeat_ngram_size,
            min_gen_length=args.min_gen_length,
        )
    else:
        raise ValueError(f"unknown strategy {args.sampling_strategy}")

    def process(raw_text):
        if args.with_id:
            query_id, raw_text = raw_text.split("\t")

        answers, answers_with_style, blanks = fill_blanks(
            raw_text, model, tokenizer, strategy, args)

        # save
        if args.with_id:
            full_path = os.path.join(args.output_path, query_id + ".txt")
        else:
            prefix = raw_text.replace("/", "")[:20]
            full_path = timed_name(prefix, ".txt", args.output_path)
        if mpu.get_model_parallel_rank() == 0:
            if args.print_all_beams and len(answers) > 1:
                for idx, answer_with_style in enumerate(answers_with_style):
                    # print the first.
                    print(f"Output beam {idx}:", answer_with_style)
                    if len(answer_with_style) > 120:
                        print("")
            else:
                print(f"Output:", answers_with_style[0])  # print the first.
            with open(full_path, "w", encoding="utf-8") as fout:
                for answer in answers:
                    fout.write(answer + "\n")

            os.chmod(full_path, stat.S_IRWXO + stat.S_IRWXG + stat.S_IRWXU)

    os.makedirs(args.output_path, exist_ok=True)
    generate_continually(process, args.input_source)
