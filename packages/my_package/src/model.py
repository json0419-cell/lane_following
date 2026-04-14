import os
import sys
from typing import Tuple
from pathlib import Path
import numpy as np

import torch
from dt_device_utils import DeviceHardwareBrand, get_device_hardware_brand

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from constants import IMAGE_SIZE, YOLO_MODEL_PATH
from ultralytics import YOLO

JETSON_FP16 = True
IMAGE_SIZE = 640

def run(input, exception_on_failure=False):
    print(input)
    try:
        import subprocess

        program_output = subprocess.check_output(
            f"{input}", shell=True, universal_newlines=True, stderr=subprocess.STDOUT
        )
    except Exception as e:
        if exception_on_failure:
            print(e.output)
            raise e
        program_output = e.output
    print(program_output)
    return program_output.strip()

class Model:
    def __init__(self, weight_file_path: str):
        super().__init__()
        weight_file_path = YOLO_MODEL_PATH
        print("--------------------------------------------------------------------------------------------------------------------------")
        print(torch.__version__)
        print(torch.cuda.is_available())
        print(torch.version.cuda)
        # 强制使用 GPU（Jetson Nano 的 GPU）
        self.device = torch.device("cuda:0")
        print(f"device == {self.device}")

        # 加载 YOLO 模型并移到 GPU
        self.model = YOLO(weight_file_path).to(self.device)

        nn_model = self.model.model
        nn_model.float()
        self.model.model.eval()

    def infer(self, image: np.ndarray) -> Tuple[list, list, list]:
        results = self.model(image, verbose=False)[0]

        if results.boxes is not None and len(results.boxes) > 0:
            xyxy = results.boxes.xyxy.cpu().numpy()
            conf = results.boxes.conf.cpu().numpy()
            clas = results.boxes.cls.cpu().numpy()
            return xyxy.tolist(), clas.tolist(), conf.tolist()

        return [], [], []
