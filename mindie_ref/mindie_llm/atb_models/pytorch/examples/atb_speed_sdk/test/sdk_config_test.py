import time

import torch
from atb_speed.common.config import atb_speed_config
from atb_speed.common.launcher import Launcher
from transformers import AutoTokenizer, AutoModelForCausalLM


class BaichuanLM(Launcher):

    def init_model(self):
        """
        模型初始化
        :return:
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True, use_fast=False)
        model = AutoModelForCausalLM.from_pretrained(self.model_path, trust_remote_code=True).half().to(self._device)
        model.eval()
        model.generation_config = self.remove_part_of_generation_config(model.generation_config)
        return model, tokenizer

    def infer(self, query):
        """
        推理代码
        :param query:
        :return:
        """
        inputs = self.tokenizer(query, return_tensors='pt')
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            start_time = time.time()
            gen_kwargs = {"max_new_tokens": 64}
            pred = self.model.generate(**inputs, **gen_kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            end_time = time.time()
            time_cost = end_time - start_time
        output = self.tokenizer.decode(pred.cpu()[0], skip_special_tokens=True)
        print(output)
        print(f"cost {time_cost}s")
        new_tokens = len(pred[0]) - len(inputs.input_ids[0])
        print(f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s")
        print(f"generate {len(output) - len(query)} new chars，({(len(output) - len(query)) / time_cost:.2f} tokens/s")
        return output

    def infer_batch(self, query):
        """
        推理代码
        :param query:
        :return:
        """
        inputs = self.tokenizer(query, return_tensors='pt', padding=True)
        inputs = inputs.to(self.model.device)
        with torch.no_grad():
            start_time = time.time()
            pred = self.model.generate(**inputs, max_new_tokens=64)
            if isinstance(pred, tuple):
                pred = pred[0]
            end_time = time.time()
            time_cost = end_time - start_time
        output = self.tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=False)
        for ind, item in enumerate(output):
            print(f"###### batch {ind} ")
            print(item)

        # print(output)
        print(f"cost {time_cost}s")
        new_tokens = len(pred[0]) - len(inputs.input_ids[0])
        print(f"generate {new_tokens} new tokens，({new_tokens / time_cost:.2f} tokens/s")
        return output


if __name__ == '__main__':
    atb_speed_config.init_config()
    baichuan = BaichuanLM(device_ids="1")
    print("---------------warm-up---------------")
    baichuan.infer('Hamlet->Shakespeare\nOne Hundred Years of Solitude->')

    print("---------------inference---------------")
    baichuan.infer('登鹳雀楼->王之涣\n夜雨寄北->')
    baichuan.infer('苹果公司的CEO是')

    query_list = ["谷歌公司的CEO是",
                  '登鹳雀楼->王之涣\n夜雨寄北->',
                  '苹果公司的CEO是',
                  '华为公司的CEO是',
                  '微软公司的CEO是']
    baichuan.infer_batch(query_list)
