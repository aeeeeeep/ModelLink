import unittest
import sys
import os
import subprocess
import glob
from pathlib import Path
import torch
from utils import ParamConfig
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
        st_file = glob.glob(os.path.join(self.config.convert_ckpt_param[9], "*.safetensors"))
        self.assertEqual(len(st_file), 4)
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
        weight_content = torch.load(os.path.join(output_dir, "mp_rank_00/model_optim_rng.pt"))
        weight_common_content = weight_content['model']['language_model'] # extract commmon content

        # embedding, encoder, output_layer is three out layers.
        self.assertEqual(len(os.listdir(output_dir)), int(self.config.convert_ckpt_param[7]))
        self.assertEqual(weight_common_content['embedding']['word_embeddings']['weight'].size(), torch.Size([32000, 3072]))
        self.assertEqual(weight_common_content['encoder']['final_norm.weight'].size(), torch.Size([3072]))

        # encoder has a common final_norm and each one has folliowing six layers
        weight_common_content['encoder'].pop('final_norm.weight')
        self.assertEqual(weight_common_content['encoder']['layers.0.self_attention.query_key_value.weight'].size(), torch.Size([1536, 3072]))
        self.assertEqual(weight_common_content['encoder']['layers.0.self_attention.dense.weight'].size(), torch.Size([3072, 512]))
        self.assertEqual(weight_common_content['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size(), torch.Size([6144, 3072]))
        self.assertEqual(weight_common_content['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size(), torch.Size([3072, 3072]))
        self.assertEqual(weight_common_content['encoder']['layers.0.input_norm.weight'].size(), torch.Size([3072]))
        self.assertEqual(weight_common_content['encoder']['layers.0.post_attention_norm.weight'].size(), torch.Size([3072]))


if __name__ == "__main__":
    unittest.main()
