import torch
import torch.nn.functional as F
import torch_npu
import ascendspeed.te.ops as aops
from torch_npu.testing.testcase import TestCase, run_tests

torch.npu.config.allow_internal_format = False

class TestSwiGlu(TestCase):

    def get_golden(self, input_self_tensor, dim):
        def swiglu_v1(x):
            """0.1版本，FP32格式运算，最后输出转成BF16"""
            x = torch.chunk(x, 2, dim=dim)
            self_tensor = x[0].type(torch.float32)
            other = x[1].type(torch.float32)
            output = F.silu(self_tensor.npu()) * other.npu()
            return output.type(torch.bfloat16)

        def swiglu_v2(x):
            """0.2版本，Silu小算子拼接的版本，最后乘法使用BF16计算"""
            x = torch.chunk(x, 2, dim=dim)
            return F.silu(x[0].npu()) * x[1].npu()

        output = swiglu_v2(input_self_tensor)
        return output

    def test_swiglu(self):
        shape = [8192, 1, 3904*2]
        dim = -1
        input_self_tensor = torch.rand(shape, device='cpu', dtype=torch.bfloat16)
        print("input:", input_self_tensor)

        golden = self.get_golden(input_self_tensor, dim)
        prof_path = "./prof_total"
        with torch.npu.profile(prof_path) as prof:
            torch.npu.synchronize()
            output = aops.swiglu(input_self_tensor.npu()).cpu()
            torch.npu.synchronize()

        print("golden:", golden)
        print("output:", output)
        self.assertRtolEqual(output.type(torch.float32), golden.type(torch.float32))


class AAATestSwiGluGrad(TestCase):

    def swish(self, beta, x):
        return x * torch.sigmoid(beta * x)

    def swish_grad(self, beta, x):
        return torch.sigmoid(beta * x) + x * (1 - torch.sigmoid(beta * x)) * torch.sigmoid(beta * x) * beta

    def get_golden(self, tensor_gradout, input_self_tensor, dim):

        def swiglu_grad_v1(x):
            """0.1版本，FP32格式运算，最后输出转成BF16"""
            beta_value = 1.0
            inTensors = torch.chunk(x, 2, dim=dim)
            tensor_self_float = inTensors[0].type(torch.float)
            tensor_other_float = inTensors[1].type(torch.float)
            tensor_gradout_float = tensor_gradout.type(torch.float)
            torch.mul(torch.relu(tensor_self_float), tensor_other_float)
            tensor_out1 = torch.mul(torch.mul(tensor_other_float, self.swish_grad(beta_value, tensor_self_float)),
                                    tensor_gradout_float)
            tensor_out2 = torch.mul(tensor_gradout_float, self.swish(beta_value, tensor_self_float))
            tensor_out_float = torch.cat((tensor_out1, tensor_out2), dim=-1)
            return tensor_out_float.type(torch.bfloat16)

        def swiglu_grad_v2(x):
            """0.2版本，Silu小算子拼接的版本，最后乘法使用BF16计算"""

            x = torch.chunk(x, 2, dim=dim)
            self_tensor = x[0].npu()
            other_tensor = x[1].npu()
            tensor_gradout_npu = tensor_gradout.npu()

            self_tensor.requires_grad_(True)
            self_tensor.retain_grad()

            swish_output = F.silu(self_tensor)
            swish_output.backward([tensor_gradout_npu])
            swish_grad = self_tensor.grad

            m = swish_grad * other_tensor
            n = tensor_gradout_npu * F.silu(self_tensor)
            output = torch.cat([m, n], dim=dim)
            return output


        output = swiglu_grad_v2(input_self_tensor)
        return output

    def aaa_test_swiglu_grad(self):
        shape = [8192, 1, 3904*2]
        grad_shape = [8192, 1, 3904]
        dim = -1
        grad_out = torch.rand(grad_shape, device='cpu', dtype=torch.bfloat16)
        input_self_tensor = torch.rand(shape, device='cpu', dtype=torch.bfloat16)
        print("input:", input_self_tensor)
        print("grad:", grad_out)

        golden = self.get_golden(grad_out, input_self_tensor, dim)
        prof_path = "./prof_total_backward"
        with torch.npu.profile(prof_path) as prof:
            torch.npu.synchronize()

            input_self_tensor_npu = input_self_tensor.npu()
            input_self_tensor_npu.requires_grad_(True)
            input_self_tensor_npu.retain_grad()
            grad_out_npu = grad_out.npu()

            output_forward = aops.swiglu(input_self_tensor_npu, dim)
            output_forward.backward([grad_out_npu])
            result = input_self_tensor_npu.grad.cpu()

            torch.npu.synchronize()

        print("golden:", golden)
        print("output:", result)
        self.assertRtolEqual(result.type(torch.float32), golden.type(torch.float32))


if __name__ == "__main__":
    run_tests()
