#!/usr/bin/env python3

import csv
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import rospy
from duckietown.dtros import DTROS, NodeType
from duckietown_msgs.msg import WheelsCmdStamped
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import Float32

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from constants import (
    HEADING_GAIN,
    HEADING_MODEL_METADATA_PATH,
    HEADING_OBSERVATION_MODE,
    HEADING_MODEL_PATH,
    HEADING_PROCESS_EVERY_N_FRAMES,
    HEADING_TYPE,
)
from heading_model import HeadingOnnxModel
from lane_mask import build_binary_lane_image_from_bgr

HEADING_CLIPPED_08 = "heading_clipped_08"
HEADING_CLIPPED_08_LIMIT = 0.8
HEADING_CLIPPED_08_DECIMALS = 2


def compressed_imgmsg_to_cv2(msg: CompressedImage) -> Optional[np.ndarray]:
    try:
        arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn("cv2.imdecode returned None; compressed image may be corrupted.")
        return image
    except Exception as exc:
        rospy.logerr(f"Failed to decode image: {exc}")
        return None


def cv2_to_compressed_imgmsg(
    cv_image: np.ndarray,
    *,
    stamp=None,
    frame_id: Optional[str] = None,
    dst_format: str = "jpeg",
) -> Optional[CompressedImage]:
    msg = CompressedImage()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    if frame_id is not None:
        msg.header.frame_id = frame_id
    ext = ".jpg" if dst_format == "jpeg" else ".png"
    success, encoded_img = cv2.imencode(ext, cv_image)
    if not success:
        rospy.logerr("cv2.imencode failed while publishing debug image.")
        return None
    msg.format = dst_format
    msg.data = encoded_img.tobytes()
    return msg


class HeadingControlNode(DTROS):
    def __init__(self, node_name: str):
        super().__init__(node_name=node_name, node_type=NodeType.PERCEPTION)

        self.vehicle_name = os.environ.get("VEHICLE_NAME", "duckiebot")
        self.model = HeadingOnnxModel(
            model_path=rospy.get_param("~model_path", HEADING_MODEL_PATH),
            metadata_path=rospy.get_param("~model_metadata_path", HEADING_MODEL_METADATA_PATH),
        )
        self.forward_speed = float(rospy.get_param("~forward_speed", self.model.forward_speed))
        self.heading_gain = float(rospy.get_param("~heading_gain", HEADING_GAIN))
        self.max_steer = float(rospy.get_param("~max_steer", self.model.max_steer))
        self.heading_type = str(rospy.get_param("~heading_type", HEADING_TYPE))
        self.observation_mode = str(rospy.get_param("~observation_mode", HEADING_OBSERVATION_MODE))
        self.publish_debug = bool(rospy.get_param("~publish_debug_image", True))
        self.command_timeout = float(rospy.get_param("~command_timeout", 0.5))
        self.process_every_n_frames = max(
            1,
            int(rospy.get_param("~process_every_n_frames", HEADING_PROCESS_EVERY_N_FRAMES)),
        )
        self.record_debug = bool(rospy.get_param("~record_debug", False))
        self.record_debug_dir = str(rospy.get_param("~record_debug_dir", "/data/heading_debug"))
        self.record_max_frames = max(0, int(rospy.get_param("~record_max_frames", 0)))

        self.camera_topic = rospy.get_param(
            "~camera_topic",
            f"/{self.vehicle_name}/camera_node/image/compressed",
        )
        self.wheels_topic = rospy.get_param(
            "~wheels_topic",
            f"/{self.vehicle_name}/wheels_driver_node/wheels_cmd",
        )

        self.frames = deque(maxlen=self.model.frame_stack)
        self.frame_counter = 0
        self.last_command_time = rospy.Time(0)
        self.record_saved_frames = 0
        self.record_root: Optional[Path] = None
        self.record_file = None
        self.record_writer = None
        self.record_raw_dir: Optional[Path] = None
        self.record_cropped_dir: Optional[Path] = None
        self.record_preprocessed_dir: Optional[Path] = None
        self.record_overlay_dir: Optional[Path] = None

        self.pub_wheels = rospy.Publisher(self.wheels_topic, WheelsCmdStamped, queue_size=1)
        self.pub_heading = rospy.Publisher("~heading", Float32, queue_size=1)
        self.pub_smooth_heading = rospy.Publisher("~smooth_heading", Float32, queue_size=1)
        self.pub_debug_image = rospy.Publisher("~image/compressed", CompressedImage, queue_size=1)

        self.sub_image = rospy.Subscriber(
            self.camera_topic,
            CompressedImage,
            self.image_cb,
            queue_size=1,
            buff_size=10_000_000,
        )
        self.watchdog = rospy.Timer(rospy.Duration(0.1), self._watchdog_cb)
        rospy.on_shutdown(self.stop_vehicle)
        rospy.on_shutdown(self._close_recorder)

        if self.record_debug:
            self._init_recorder()

        self.log(
            f"Loaded ONNX model from {self.model.config['model_path']} "
            f"with providers {self.model.session.get_providers()}"
        )
        if "CUDAExecutionProvider" in self.model.session.get_providers():
            self.log("Using GPU via ONNX Runtime CUDAExecutionProvider")
        else:
            self.log("Using CPU via ONNX Runtime")
        self.log(
            "Control config: "
            f"heading_type={self.heading_type} "
            f"observation_mode={self.observation_mode} "
            f"forward_speed={self.forward_speed:.3f} "
            f"heading_gain={self.heading_gain:.3f} "
            f"max_steer={self.max_steer:.3f} "
            f"process_every_n_frames={self.process_every_n_frames} "
            f"record_debug={self.record_debug} "
            f"record_max_frames={self.record_max_frames}"
        )
        self.log(f"Subscribing to {self.camera_topic}")
        self.log(f"Publishing wheel commands to {self.wheels_topic}")
        if self.record_root is not None:
            self.log(f"Recording debug frames to {self.record_root}")

    def _init_recorder(self) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.record_root = Path(self.record_debug_dir) / timestamp
        self.record_raw_dir = self.record_root / "raw"
        self.record_cropped_dir = self.record_root / "cropped"
        self.record_preprocessed_dir = self.record_root / "preprocessed"
        self.record_overlay_dir = self.record_root / "overlay"
        for path in (
            self.record_raw_dir,
            self.record_cropped_dir,
            self.record_preprocessed_dir,
            self.record_overlay_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

        csv_path = self.record_root / "sequence.csv"
        self.record_file = csv_path.open("w", newline="", encoding="utf-8")
        self.record_writer = csv.DictWriter(
            self.record_file,
            fieldnames=[
                "frame_idx",
                "stamp",
                "raw_heading",
                "final_heading",
                "left",
                "right",
                "raw_path",
                "cropped_path",
                "preprocessed_path",
                "overlay_path",
            ],
        )
        self.record_writer.writeheader()
        self.record_file.flush()

    def _close_recorder(self) -> None:
        if self.record_file is not None:
            try:
                self.record_file.close()
            except Exception:
                pass
            self.record_file = None

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        top = int(bgr.shape[0] * self.model.crop_top_ratio)
        cropped_bgr = bgr[top:, :, :]
        resized_bgr = cv2.resize(
            cropped_bgr,
            (self.model.resize_width, self.model.resize_height),
            interpolation=cv2.INTER_AREA,
        )
        if self.observation_mode == "binary_lane":
            preprocessed = build_binary_lane_image_from_bgr(resized_bgr)
            if self.record_debug:
                cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            else:
                cropped = resized_bgr
        elif self.observation_mode == "rgb":
            cropped = cv2.cvtColor(cropped_bgr, cv2.COLOR_BGR2RGB)
            preprocessed = cv2.cvtColor(resized_bgr, cv2.COLOR_BGR2RGB)
        else:
            raise ValueError(f"Unsupported observation_mode={self.observation_mode}")
        normalized = preprocessed.astype(np.float32) / 255.0
        return cropped, preprocessed, normalized

    def _stack_observation(self, frame: np.ndarray) -> np.ndarray:
        if not self.frames:
            for _ in range(self.model.frame_stack):
                self.frames.append(frame.copy())
        else:
            self.frames.append(frame.copy())
        return np.concatenate(list(self.frames), axis=2).astype(np.float32)

    def _heading_to_wheels(self, heading_action: float) -> Tuple[float, float, float]:
        heading = float(np.clip(heading_action, -1.0, 1.0))
        if self.heading_type == "heading_smooth":
            heading = (heading ** 3) * self.max_steer
        elif self.heading_type == HEADING_CLIPPED_08:
            heading = float(np.clip(heading, -HEADING_CLIPPED_08_LIMIT, HEADING_CLIPPED_08_LIMIT))
            heading = float(np.round(heading, HEADING_CLIPPED_08_DECIMALS))
            heading = float(np.clip(heading * self.max_steer, -HEADING_CLIPPED_08_LIMIT, HEADING_CLIPPED_08_LIMIT))
            heading = float(np.round(heading, HEADING_CLIPPED_08_DECIMALS))
        else:
            heading = heading * self.max_steer

        heading *= self.heading_gain

        wheels = np.array([1.0 + heading, 1.0 - heading], dtype=np.float32)
        wheels = np.clip(wheels, 0.0, 1.0)
        if self.forward_speed != 1.0:
            wheels = np.clip(wheels * self.forward_speed, 0.0, 1.0)
        return float(wheels[0]), float(wheels[1]), float(heading)

    def _publish_wheels(self, left: float, right: float, stamp=None) -> None:
        msg = WheelsCmdStamped()
        msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
        msg.vel_left = float(left)
        msg.vel_right = float(right)
        self.pub_wheels.publish(msg)
        self.last_command_time = msg.header.stamp

    def _make_overlay_image(
        self,
        preprocessed_rgb: np.ndarray,
        raw_heading: float,
        final_heading: float,
        left: float,
        right: float,
    ) -> np.ndarray:
        overlay_bgr = cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR)
        lines = [
            f"raw_heading={raw_heading:+.3f}",
            f"final_heading={final_heading:+.3f}",
            f"left={left:.3f} right={right:.3f}",
        ]
        for idx, text in enumerate(lines):
            cv2.putText(
                overlay_bgr,
                text,
                (4, 18 + idx * 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return overlay_bgr

    def _record_step(
        self,
        *,
        image_msg: CompressedImage,
        raw_bgr: np.ndarray,
        cropped_rgb: np.ndarray,
        preprocessed_rgb: np.ndarray,
        overlay_bgr: np.ndarray,
        raw_heading: float,
        final_heading: float,
        left: float,
        right: float,
    ) -> None:
        if not self.record_debug or self.record_writer is None:
            return
        if self.record_max_frames > 0 and self.record_saved_frames >= self.record_max_frames:
            return

        frame_idx = self.record_saved_frames + 1
        filename = f"{frame_idx:06d}.jpg"
        raw_path = self.record_raw_dir / filename
        cropped_path = self.record_cropped_dir / filename
        preprocessed_path = self.record_preprocessed_dir / filename
        overlay_path = self.record_overlay_dir / filename

        cv2.imwrite(str(raw_path), raw_bgr)
        cv2.imwrite(str(cropped_path), cv2.cvtColor(cropped_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(preprocessed_path), cv2.cvtColor(preprocessed_rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(overlay_path), overlay_bgr)

        self.record_writer.writerow(
            {
                "frame_idx": frame_idx,
                "stamp": image_msg.header.stamp.to_sec(),
                "raw_heading": raw_heading,
                "final_heading": final_heading,
                "left": left,
                "right": right,
                "raw_path": f"raw/{raw_path.name}",
                "cropped_path": f"cropped/{cropped_path.name}",
                "preprocessed_path": f"preprocessed/{preprocessed_path.name}",
                "overlay_path": f"overlay/{overlay_path.name}",
            }
        )
        self.record_file.flush()
        self.record_saved_frames = frame_idx

        if self.record_max_frames > 0 and self.record_saved_frames == self.record_max_frames:
            self.log(f"Reached record_max_frames={self.record_max_frames}; stopping debug capture.")

    def _watchdog_cb(self, _event) -> None:
        if self.command_timeout <= 0.0 or self.last_command_time == rospy.Time(0):
            return
        elapsed = (rospy.Time.now() - self.last_command_time).to_sec()
        if elapsed > self.command_timeout:
            self._publish_wheels(0.0, 0.0)
            self.last_command_time = rospy.Time.now()

    def image_cb(self, image_msg: CompressedImage) -> None:
        self.frame_counter += 1
        if (self.frame_counter - 1) % self.process_every_n_frames != 0:
            return

        bgr = compressed_imgmsg_to_cv2(image_msg)
        if bgr is None:
            self._publish_wheels(0.0, 0.0)
            return

        cropped_rgb, preprocessed_rgb, normalized = self._preprocess(bgr)
        stacked = self._stack_observation(normalized)

        try:
            heading_action = self.model.predict_heading(stacked)
        except Exception as exc:
            rospy.logerr_throttle(1.0, f"ONNX inference failed: {exc}")
            self._publish_wheels(0.0, 0.0)
            return

        left, right, smooth_heading = self._heading_to_wheels(heading_action)
        self._publish_wheels(left, right, stamp=image_msg.header.stamp)
        self.pub_heading.publish(Float32(data=float(heading_action)))
        self.pub_smooth_heading.publish(Float32(data=float(smooth_heading)))
        rospy.loginfo_throttle(
            1.0,
            (
                f"raw_heading={heading_action:+.4f} "
                f"smooth_heading={smooth_heading:+.4f} "
                f"left={left:.3f} right={right:.3f}"
            ),
        )

        overlay_bgr = self._make_overlay_image(
            preprocessed_rgb,
            heading_action,
            smooth_heading,
            left,
            right,
        )
        self._record_step(
            image_msg=image_msg,
            raw_bgr=bgr,
            cropped_rgb=cropped_rgb,
            preprocessed_rgb=preprocessed_rgb,
            overlay_bgr=overlay_bgr,
            raw_heading=heading_action,
            final_heading=smooth_heading,
            left=left,
            right=right,
        )

        if self.publish_debug and self.pub_debug_image.get_num_connections() > 0:
            debug_msg = cv2_to_compressed_imgmsg(
                overlay_bgr,
                stamp=image_msg.header.stamp,
                frame_id=image_msg.header.frame_id,
            )
            if debug_msg is not None:
                self.pub_debug_image.publish(debug_msg)

    def stop_vehicle(self) -> None:
        try:
            self._publish_wheels(0.0, 0.0)
        except Exception:
            pass


if __name__ == "__main__":
    node = HeadingControlNode(node_name="heading_control_node")
    rospy.spin()
