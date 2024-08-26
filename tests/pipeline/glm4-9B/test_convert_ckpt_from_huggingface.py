import unittest
import sys
import os
import subprocess
import glob
from pathlib import Path
from utils import ParamConfig
import torch

import modellink


class TestConvertCkptFromHuggingface(unittest.TestCase):
    def setUp(self, config=ParamConfig):
        # configure params, the index starts from 1
        self.config = config
        sys.argv = [sys.argv[0]] + self.config.convert_ckpt_param
    
    def test_file_exsit(self):
        """
        Test if the file in the `--load-dir` exsit, including `.bin`, `.json`...
        """
        satetensors_file = glob.glob(os.path.join(self.config.convert_ckpt_param[9], "*.safetensors"))
        self.assertEqual(len(satetensors_file), 10)
        self.assertTrue(os.path.exists(os.path.join(self.config.convert_ckpt_param[9], "model.safetensors.index.json")))
    
    def test_convert_weights_form_huggingface(self):
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        base_dir = Path(__file__).absolute().parent.parent.parent.parent
        file_path = os.path.join(base_dir, "tools/checkpoint/convert_ckpt.py")
        arguments = sys.argv[1:]
        subprocess.run(["python", file_path] + arguments)
        output_dir = os.path.join(self.config.convert_ckpt_param[11], "iter_0000001")
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00_000/model_optim_rng.pt"))
        weight_common_content = weight_content['model'] # extract commmon content

        # embedding, encoder, output_layer is three out layers.
        self.assertEqual(len(os.listdir(output_dir)), int(self.config.convert_ckpt_param[13]) * int(self.config.convert_ckpt_param[15]))
        self.assertEqual(weight_common_content['embedding.word_embeddings.weight'].size(), torch.Size([75776, 4096]))

        # encoder has a common final_norm and each one has folliowing six layers
        self.assertEqual(weight_common_content['decoder.layers.1.self_attention.linear_qkv.weight'].size(), torch.Size([2304, 4096]))
        self.assertEqual(weight_common_content['decoder.layers.1.self_attention.linear_proj.weight'].size(), torch.Size([4096, 2048]))
        self.assertEqual(weight_common_content['decoder.layers.0.mlp.linear_fc1.weight'].size(), torch.Size([13696, 4096]))
        self.assertEqual(weight_common_content['decoder.layers.0.mlp.linear_fc2.weight'].size(), torch.Size([4096, 6848]))
        self.assertEqual(weight_common_content['decoder.layers.0.input_layernorm.weight'].size(), torch.Size([4096]))
        self.assertEqual(weight_common_content['decoder.layers.0.pre_mlp_layernorm.weight'].size(), torch.Size([4096]))


if __name__ == "__main__":
    unittest.main()
