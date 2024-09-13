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
    
    
    def test_model_file_exist(self):
        """
        Test if the model file in the `--load-dir` exist, including `.bin`, `.json`...
        """
        load_dir_idx = self.config.convert_ckpt_param.index('--load-dir') + 1
        bin_file_num = 7
        bin_file = glob.glob(os.path.join(self.config.convert_ckpt_param[load_dir_idx], "*.bin"))
        json_file = "pytorch_model.bin.index.json"
        json_path = os.path.join(self.config.convert_ckpt_param[load_dir_idx], json_file)
        self.assertEqual(len(bin_file), bin_file_num)
        self.assertTrue(os.path.exists(json_path), json_path + "does not exist")
    

    def test_convert_weights_from_huggingface(self):
        """
        Test whether the weight to be converted as we want in `--save-dir`. We will check the model layer name, 
        including embedding, final_norm, output and encoder. In the encoder, there will be some different layers 
        to compose the unique transformer layer and all these layer stack to compose the entity of the model.
        """
        # run convert weight
        base_dir = Path(__file__).absolute().parent.parent.parent.parent
        file_path = os.path.join(base_dir, "convert_ckpt.py")
        arguments = sys.argv[1:]
        subprocess.run(["python", file_path] + arguments)

        # save_dir file count
        save_dir_idx = self.config.convert_ckpt_param.index('--save-dir') + 1
        output_dir = os.path.join(self.config.convert_ckpt_param[save_dir_idx], "iter_0000001")
        self.assertEqual(len(os.listdir(output_dir)), int(self.config.convert_ckpt_param[9]))

        # word_embedding count
        word_embedding_content = torch.load(os.path.join(output_dir, "mp_rank_00_000/model_optim_rng.pt"))
        we_weight = word_embedding_content['model']['language_model']
        self.assertEqual(we_weight['embedding']['word_embeddings']['weight'].size(), torch.Size([65024, 4096]))
        
        # common weight count
        common_content = torch.load(os.path.join(output_dir, "mp_rank_00_001/model_optim_rng.pt"))
        model_weight = common_content['model']['language_model']
        # encoder has a common final_norm and each one has folliowing 14 layers
        self.assertEqual(model_weight['encoder']['final_norm.weight'].size(), torch.Size([4096]))
        model_weight['encoder'].pop('final_norm.weight')
        self.assertEqual(model_weight['encoder']['layers.0.self_attention.query_key_value.weight'].size(), torch.Size([4608, 4096]))
        self.assertEqual(model_weight['encoder']['layers.0.self_attention.query_key_value.bias'].size(), torch.Size([4608]))
        self.assertEqual(model_weight['encoder']['layers.0.self_attention.dense.weight'].size(), torch.Size([4096, 4096]))
        self.assertEqual(model_weight['encoder']['layers.0.mlp.dense_h_to_4h.weight'].size(), torch.Size([27392, 4096]))
        self.assertEqual(model_weight['encoder']['layers.0.mlp.dense_4h_to_h.weight'].size(), torch.Size([4096, 13696]))
        self.assertEqual(model_weight['encoder']['layers.0.input_norm.weight'].size(), torch.Size([4096]))
        self.assertEqual(model_weight['encoder']['layers.0.post_attention_norm.weight'].size(), torch.Size([4096]))
        
        # output_layer count
        self.assertEqual(model_weight['output_layer']['weight'].size(), torch.Size([65024, 4096]))


if __name__ == "__main__":
    unittest.main()
