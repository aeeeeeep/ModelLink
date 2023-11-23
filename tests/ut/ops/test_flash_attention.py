import unittest
from unittest import TestCase
import torch
import torch_npu
import ascendspeed_te_ops
import numpy as np

atten_mask_value = -3e30
torch.npu.set_device(5)


class AscendTeOpTest(TestCase):

    def torch_batch_dot(self, a, b):
        batch, n, h, _ = a.shape
        _, _, _, w = b.shape
        a = a.to(torch.float)
        b = b.to(torch.float)
        res = torch.zeros((batch, n, h, w), dtype=torch.float)
        for i in range(batch):
            for j in range(n):
                res[i, j] = torch.matmul(a[i, j], b[i, j])
                
        return res
    
    def compare_result(self, result_label, result_temp, limit_error_ratio, relative_error_limit=0.001, absolute_error_limit=0.001, error_tag=""):
        print("==============start compare {}=================".format(error_tag))
        ori_shape = result_label.shape
        result_label = result_label.reshape(-1).astype("float32")
        result_temp = result_temp.reshape(-1).astype("float32")
        error_mask = (np.abs(result_label - result_temp) / (np.abs(result_label) + 0.00000000001)) > relative_error_limit
        error_num = np.sum(error_mask)

        error_ratio = error_num / result_temp.size
        print("相对误差 {} {} error 数量 {} error 比例 {}".format(error_tag, relative_error_limit, error_num, error_ratio))

        # assert operation
        # self.assertEqual(error_num, 0)
        self.assertGreaterEqual(limit_error_ratio, error_ratio)

        print("error data")
        error_mask_1 = np.abs(result_label - result_temp) > absolute_error_limit
        print(result_label[error_mask & error_mask_1][:10])
        print(result_temp[error_mask & error_mask_1][:10])
        error_mask = error_mask.reshape(ori_shape)
        error_mask_1 = error_mask_1.reshape(ori_shape)
        print("error index:", np.argwhere(error_mask & error_mask_1 > 0))
        print("error max:", np.max(np.abs(result_label - result_temp)))
        print("==============end compare {}=================".format(error_tag))

    def attention_ori(self, query, key, value, atten_mask, scale_value, dtype):
        """
        (b, n, s,  d)
        """
        query = query.to(dtype)
        key = key.to(dtype)
        value = value.to(dtype)

        bmm1_res = self.torch_batch_dot(key, torch.transpose(query, 2, 3)).to(dtype).to(torch.float)
        bmm1_res = bmm1_res * scale_value + (atten_mask * atten_mask_value)
        softmax_max, _ = torch.max(bmm1_res, dim=-2, keepdim=True)
        softmax_sub = bmm1_res - softmax_max
        softmax_exp = torch.exp(softmax_sub)
        softmax_sum = torch.sum(softmax_exp, dim=-2, keepdim=True)
        softmax_out = softmax_exp / softmax_sum

        softmax_out = softmax_out.to(dtype).to(torch.float)
        attention_out = self.torch_batch_dot(torch.transpose(softmax_out, 2, 3), value)

        softmax_log_max_sum = softmax_max + torch.log(softmax_sum)
        return softmax_max, softmax_sum, softmax_log_max_sum, attention_out


    def attention_grad_ori(self, query, key, value, atten_mask, softmax_log_max_sum, attention_score, attention_score_grad, scale_value, dtype):
        """
        (b, n, s,  d)
        """
        batch_size, head_num, seq_size, head_dim = query.shape

        query = query.to(dtype)
        key = key.to(dtype)
        value = value.to(dtype)
        attention_score = attention_score.to(dtype)
        attention_score_grad = attention_score_grad.to(dtype)

        softmax_log_max_sum = softmax_log_max_sum.view((batch_size, head_num, 1, seq_size))

        softmax_out_sum = (attention_score.to(torch.float) * attention_score_grad.to(torch.float)).sum(axis=-1).view((batch_size, head_num, 1, seq_size))

        query_key_mul = self.torch_batch_dot(key, torch.transpose(query, 2, 3)).to(dtype).to(torch.float)
        softmax_grad = self.torch_batch_dot(value, torch.transpose(attention_score_grad, 2, 3)).to(dtype).to(torch.float)

        bmm1_res_drop = query_key_mul * scale_value + (atten_mask * atten_mask_value)
        softmax_out = torch.exp(bmm1_res_drop - softmax_log_max_sum)

        bmm1_res_drop_grad_flash = (softmax_grad - softmax_out_sum) * softmax_out
        bmm1_res_grad = bmm1_res_drop_grad_flash * scale_value

        value_grad = self.torch_batch_dot(softmax_out.to(dtype), attention_score_grad)
        key_grad = self.torch_batch_dot(bmm1_res_grad.to(dtype), query)
        query_grad = self.torch_batch_dot(torch.transpose(bmm1_res_grad.to(dtype), 2, 3), key)
        return query_grad, key_grad, value_grad
    
    def get_data(self, input_shape, batch_size, head_num, seq_size, head_dim, dtype, scale_value, seed, input_layout, keep_prob, pre_tokens, next_tokens):
        attention_score_grad = (torch.rand(input_shape).to(torch.float) - 0.5)
        query = (torch.rand(input_shape).to(torch.float) - 0.5) * 5
        key = (torch.rand(input_shape).to(torch.float) - 0.5) * 5
        value = (torch.rand(input_shape).to(torch.float) - 0.5) * 5
        
        atten_mask = np.tri(seq_size, k=-next_tokens) + np.tri(seq_size, k=-pre_tokens).transpose()
        alibi_mask = torch.zeros((batch_size, head_num, seq_size, seq_size)).to(torch.float)
        atten_mask = torch.from_numpy(atten_mask).to(torch.float)

        softmax_max, softmax_sum, softmax_log_max_sum, attention_score = self.attention_ori(query, key, value, atten_mask, scale_value, dtype=dtype)

        query_grad, key_grad, value_grad = self.attention_grad_ori(query, key, value, atten_mask, softmax_log_max_sum, attention_score, attention_score_grad, scale_value, dtype=dtype)
        return attention_score_grad, query, key, value, atten_mask, softmax_log_max_sum, attention_score, query_grad, key_grad, value_grad
    

    def faAll(self, input_shape, softmax_shape, seq_size, head_num, scale_value, input_layout, keep_prob, pre_tokens, next_tokens, dtype):
        batch_size, head_num, seq_size, head_dim = input_shape
        seed = 2
        attention_score_grad, query, key, value, atten_mask, softmax_log_max_sum, attention_score, query_grad, key_grad, value_grad = self.get_data(input_shape, batch_size, head_num, seq_size, head_dim, dtype, scale_value, seed, input_layout, keep_prob, pre_tokens, next_tokens)
        if input_layout == "BNSD":
            pass
        elif input_layout == "BSH":
            data_shape = (batch_size, seq_size, head_num * head_dim)
            transpose_dim = (0, 2, 1, 3)
            query = query.to(dtype).permute(transpose_dim).reshape(data_shape)
            key = key.to(dtype).permute(transpose_dim).reshape(data_shape)
            value = value.to(dtype).permute(transpose_dim).reshape(data_shape)

            query_grad_label = query_grad.to(dtype).to(torch.float).numpy().transpose(transpose_dim).reshape(data_shape)
            key_grad_label = key_grad.to(dtype).to(torch.float).numpy().transpose(transpose_dim).reshape(data_shape)
            value_grad_label = value_grad.to(dtype).to(torch.float).numpy().transpose(transpose_dim).reshape(data_shape)
        else:
            data_shape = (seq_size, batch_size, head_num * head_dim)
            transpose_dim = (2, 0, 1, 3)
            query = query.to(dtype).permute(transpose_dim).reshape(data_shape)
            key = key.to(dtype).permute(transpose_dim).reshape(data_shape)
            value = value.to(dtype).permute(transpose_dim).reshape(data_shape)

            query_grad_label = query_grad.to(dtype).to(torch.float).numpy().transpose(transpose_dim).reshape(data_shape)
            key_grad_label = key_grad.to(dtype).to(torch.float).numpy().transpose(transpose_dim).reshape(data_shape)
            value_grad_label = value_grad.to(dtype).to(torch.float).numpy().transpose(transpose_dim).reshape(data_shape)
        attention_score_label = attention_score
        softmax_log_max_sum_label = softmax_log_max_sum

        query.requires_grad = True
        key.requires_grad = True
        value.requires_grad = True
        result = ascendspeed_te_ops.ascend_flash_attention(query.to(dtype).npu(), key.to(dtype).npu(), value.to(dtype).npu(), atten_mask.to(dtype).npu(), None, None, scale_value, head_num, 0, keep_prob, pre_tokens, next_tokens, 0, -1)
        
        result[1].backward()
        softmax_log_max_sum = result[0].cpu().to(torch.float).numpy()
        attention_score = result[1].cpu().to(torch.float).numpy()
        query_grad = query.grad.data.cpu().to(torch.float).numpy()
        key_grad = key.grad.data.cpu().to(torch.float).numpy()
        value_grad = value.grad.data.cpu().to(torch.float).numpy()

        return attention_score_label.to(torch.float).numpy(), attention_score, softmax_log_max_sum_label.to(torch.float).numpy(), softmax_log_max_sum, query_grad_label.to(torch.float).numpy(), query_grad, key_grad_label.to(torch.float).numpy(), key_grad, value_grad_label.to(torch.float).numpy(), value_grad
        

    def test_fa(self):
        seed = 2
        batch_size, head_num, seq_size, head_dim = 1, 2, 512, 128
        input_layout = "BNSD"  # BSH  SBH
        keep_prob = 1.0
        pre_tokens = 65536
        next_tokens = 1
        scale_value = 0.088
        
        input_shape = (batch_size, head_num, seq_size, head_dim)
        softmax_shape = (batch_size, head_num, seq_size)
        dtype = torch.float16
        torch.npu.manual_seed(seed)
        attention_score_label, attention_score, softmax_log_max_sum_label, softmax_log_max_sum, query_grad_label, query_grad, key_grad_label, key_grad, value_grad_label, value_grad = self.faAll(input_shape, softmax_shape, seq_size, head_num, scale_value, input_layout, keep_prob, pre_tokens, next_tokens, dtype)
        self.attention_score_label = attention_score_label
        # 正向
        self.compare_result(attention_score_label, attention_score, limit_error_ratio = 0.01, relative_error_limit = 0.1, absolute_error_limit = 0.1, error_tag="attention_score_label")
        self.compare_result(softmax_log_max_sum_label, softmax_log_max_sum, limit_error_ratio = 0.01, relative_error_limit = 0.1, absolute_error_limit = 0.1, error_tag="softmax_log_max_sum_label")
        # 反向
        self.compare_result(query_grad_label, query_grad, limit_error_ratio = 0.01, relative_error_limit = 0.1, absolute_error_limit = 0.1, error_tag="query_grad_label")
        self.compare_result(key_grad_label, key_grad, limit_error_ratio = 0.01, relative_error_limit = 0.1, absolute_error_limit = 0.1, error_tag="key_grad_label")
        self.compare_result(value_grad_label, value_grad, limit_error_ratio = 0.01, relative_error_limit = 0.1, absolute_error_limit = 0.1, error_tag="value_grad_label")


if __name__ == '__main__':
    unittest.main(buffer=False)
