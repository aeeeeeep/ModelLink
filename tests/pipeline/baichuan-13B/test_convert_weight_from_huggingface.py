import unittest
import sys
import os
import subprocess
import glob
from pathlib import Path
import torch
from utils import ParamConfig

from ascendspeed import megatron_adaptor
import modellink


class TestConvertCkptFromHuggingface(unittest.TestCase):
    def setUp(self):
        # configure params, the index starts from 1
        sys.argv = [sys.argv[0]] + ParamConfig.convert_weight_param

    def test_file_exsit(self):
        """
        Test if the file in the `--load-dir` exsit, including `.bin`, `.json`...
        """
        bin_file = glob.glob(os.path.join(sys.argv[10], "*.bin"))
        self.assertEqual(len(bin_file), 3)
        self.assertTrue(os.path.exists(os.path.join(sys.argv[10], "pytorch_model.bin.index.json")))

    def test_convert_weights_form_huggingface(self):
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parent.parent.parent.parent
        file_path = os.path.join(base_dir, "tools/checkpoint/util.py")
        arguments = sys.argv[1:]
        subprocess.run(["python", file_path] + arguments)
        output_dir = os.path.join(sys.argv[12], "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_common_content = weight_content['model']['language_model']  # extract commmon content

        # embedding, encoder, output_layer is three out layers.
        self.assertEqual(len(os.listdir(output_dir)), int(sys.argv[8]))
        self.assertEqual(weight_common_content['embedding']['word_embeddings']['weight'].size(),
                         torch.Size([8000, 5120]))
        self.assertEqual(weight_common_content['encoder']['final_norm.weight'].size(), torch.Size([5120]))

        # encoder has a common final_norm and each one has folliowing six layers
        weight_common_content['encoder'].pop('final_norm.weight')
        self.assertEqual(len(weight_common_content['encoder']) / 40, 6)
        self.assertEqual(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size(),
                         torch.Size([1920, 5120]))
        self.assertEqual(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size(),
                         torch.Size([5120, 640]))
        self.assertEqual(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size(),
                         torch.Size([3424, 5120]))
        self.assertEqual(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size(),
                         torch.Size([5120, 1712]))
        self.assertEqual(weight_common_content['encoder']['layers.0.input_norm.weight'].size(), torch.Size([5120]))
        self.assertEqual(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size(),
                         torch.Size([5120]))

        self.assertEqual(weight_common_content['output_layer']['weight'].size(), torch.Size([8000, 5120]))


if __name__ == "__main__":
    unittest.main()
