import json
import os.path as osp


ALPACA_TEMPLATE = {
    "system_token": "",
    "user_token": "### Instruction:",
    "assistant_token": "### Response:",
    "end_token": "",
    "system": "Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request. Please note that you need to think through your response logically and step by step."
}


class Prompter(object):
    __slots__ = ("template", "_verbose")

    def __init__(self, template, verbose: bool = False):
        self._verbose = verbose
        self.template = template
    
    def generate_training_prompt(self, messages) -> str:
        prompt = self.template["system_token"] + "\n" + self.template["system"] + self.template["end_token"] + "\n"

        for message in messages:
            if message["role"] == "user":
                prompt += self.template["user_token"] + "\n" + message["content"] + self.template["end_token"] + "\n"
            else:
                prompt += self.template["assistant_token"] + "\n" + message["content"] + self.template["end_token"] + "\n"
            
        return prompt