import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from constants import (
    HEADING_CROP_TOP_RATIO,
    HEADING_FORWARD_SPEED,
    HEADING_FRAME_STACK,
    HEADING_MAX_STEER,
    HEADING_MODEL_METADATA_PATH,
    HEADING_MODEL_PATH,
    HEADING_OBS_HEIGHT,
    HEADING_OBS_WIDTH,
    HEADING_TYPE,
)


def _default_config() -> dict:
    return {
        "model_path": HEADING_MODEL_PATH,
        "metadata_path": HEADING_MODEL_METADATA_PATH,
        "crop_top_ratio": HEADING_CROP_TOP_RATIO,
        "resize_hw": [HEADING_OBS_HEIGHT, HEADING_OBS_WIDTH],
        "frame_stack": HEADING_FRAME_STACK,
        "forward_speed": HEADING_FORWARD_SPEED,
        "max_steer": HEADING_MAX_STEER,
        "heading_type": HEADING_TYPE,
    }


class HeadingOnnxModel:
    def __init__(self, model_path: Optional[str] = None, metadata_path: Optional[str] = None):
        self.config = _default_config()
        self.config["model_path"] = model_path or self.config["model_path"]
        self.config["metadata_path"] = metadata_path or self.config["metadata_path"]

        metadata_file = Path(self.config["metadata_path"])
        if metadata_file.exists():
            metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
            resize_hw = metadata.get("resize_hw", self.config["resize_hw"])
            self.config["crop_top_ratio"] = float(metadata.get("crop_top_ratio", self.config["crop_top_ratio"]))
            self.config["frame_stack"] = int(metadata.get("frame_stack", self.config["frame_stack"]))
            self.config["forward_speed"] = float(metadata.get("forward_speed", self.config["forward_speed"]))
            self.config["max_steer"] = float(metadata.get("max_steer", self.config["max_steer"]))
            self.config["heading_type"] = str(metadata.get("heading_type", self.config["heading_type"]))
            self.config["resize_hw"] = [int(resize_hw[0]), int(resize_hw[1])]

        available = ort.get_available_providers()
        providers = [provider for provider in ("CUDAExecutionProvider", "CPUExecutionProvider") if provider in available]
        if not providers:
            raise RuntimeError(f"No supported ONNX Runtime providers found. Available providers: {available}")

        self.session = ort.InferenceSession(self.config["model_path"], providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    @property
    def crop_top_ratio(self) -> float:
        return float(self.config["crop_top_ratio"])

    @property
    def frame_stack(self) -> int:
        return int(self.config["frame_stack"])

    @property
    def resize_height(self) -> int:
        return int(self.config["resize_hw"][0])

    @property
    def resize_width(self) -> int:
        return int(self.config["resize_hw"][1])

    @property
    def forward_speed(self) -> float:
        return float(self.config["forward_speed"])

    @property
    def max_steer(self) -> float:
        return float(self.config["max_steer"])

    @property
    def heading_type(self) -> str:
        return str(self.config["heading_type"])

    def predict_heading(self, stacked_obs: np.ndarray) -> float:
        if stacked_obs.ndim != 3:
            raise ValueError(f"Expected HWC stacked observation, got shape={stacked_obs.shape}")
        batch = np.expand_dims(np.ascontiguousarray(stacked_obs, dtype=np.float32), axis=0)
        heading = self.session.run([self.output_name], {self.input_name: batch})[0]
        return float(np.asarray(heading, dtype=np.float32).reshape(-1)[0])
