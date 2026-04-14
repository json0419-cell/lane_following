#!/usr/bin/env python3

import os
import sys
from collections import deque
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

from constants import HEADING_GAIN, HEADING_MODEL_METADATA_PATH, HEADING_MODEL_PATH, HEADING_PROCESS_EVERY_N_FRAMES
from heading_model import HeadingOnnxModel


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
        self.heading_type = str(rospy.get_param("~heading_type", self.model.heading_type))
        self.publish_debug = bool(rospy.get_param("~publish_debug_image", True))
        self.command_timeout = float(rospy.get_param("~command_timeout", 0.5))
        self.process_every_n_frames = max(
            1,
            int(rospy.get_param("~process_every_n_frames", HEADING_PROCESS_EVERY_N_FRAMES)),
        )

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
            f"forward_speed={self.forward_speed:.3f} "
            f"heading_gain={self.heading_gain:.3f} "
            f"max_steer={self.max_steer:.3f} "
            f"process_every_n_frames={self.process_every_n_frames}"
        )
        self.log(f"Subscribing to {self.camera_topic}")
        self.log(f"Publishing wheel commands to {self.wheels_topic}")

    def _preprocess(self, bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        top = int(rgb.shape[0] * self.model.crop_top_ratio)
        cropped = rgb[top:, :, :]
        resized = cv2.resize(
            cropped,
            (self.model.resize_width, self.model.resize_height),
            interpolation=cv2.INTER_AREA,
        )
        normalized = resized.astype(np.float32) / 255.0
        return resized, normalized

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

        debug_rgb, normalized = self._preprocess(bgr)
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

        if self.publish_debug and self.pub_debug_image.get_num_connections() > 0:
            debug_bgr = cv2.cvtColor(debug_rgb, cv2.COLOR_RGB2BGR)
            overlay = [
                f"raw_heading={heading_action:+.3f}",
                f"smooth_heading={smooth_heading:+.3f}",
                f"left={left:.3f} right={right:.3f}",
            ]
            for idx, text in enumerate(overlay):
                cv2.putText(
                    debug_bgr,
                    text,
                    (4, 18 + idx * 18),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (0, 255, 0),
                    1,
                    cv2.LINE_AA,
                )
            debug_msg = cv2_to_compressed_imgmsg(
                debug_bgr,
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
