# 该脚本运行请使用torch==1.13.0, GPU环境 
import os 
import argparse

from PIL import Image
import numpy as np
import torch
from transformers import AutoModel


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="load Model weights and run.")
    parser.add_argument(
        "--model_path",
        type=str,
        default = "/home/x30033355/testdata/weights/visualglm6b",
        help="Location of Model weights, which contains model folders",
        )
    args = parser.parse_args()

    image_path = os.path.join(args.model_path, "examples/1.jpeg")
    onnx_model_dir = os.path.join(args.model_path, 'transfer_model')
    if not os.path.exists(onnx_model_dir):
        os.makedirs(onnx_model_dir)
    
    onnx_model_path = os.path.join(onnx_model_dir, "blip2.onnx")
    print('onnx_model_path:', onnx_model_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModel.from_pretrained(args.model_path, trust_remote_code=True).half().cuda()

    image = Image.open(image_path)
    image = image.resize((224, 224))
    image = np.array(image).astype(np.float16)
    image = torch.from_numpy(image).unsqueeze(0).to(device)
    print('input size:', image.size())  # input size: torch.Size([1, 224, 224, 3])

    torch.onnx.export(model,                   # model being run
                    image,                     # model input (or a tuple for multiple inputs)
                    onnx_model_path,           # where to save the model (can be a file or file-like object)
                    export_params=True,        # store the trained parameter weights inside the model file
                    opset_version=13,          # the ONNX version to export the model to
                    do_constant_folding=False, # whether to execute constant folding for optimization
                    input_names=['input'],     # the model's input names
                    output_names=['output'],   # the model's output names
                    dynamic_axes={'input': {0: 'batch'}},)