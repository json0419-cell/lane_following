#!/usr/bin/env python3

import cv2
import numpy as np
import rospy
import os
import sys
from pathlib import Path
from duckietown.dtros import DTROS, NodeType, TopicType
from duckietown_msgs.msg import Twist2DStamped, EpisodeStart
from sensor_msgs.msg import CompressedImage
from typing import Tuple

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from model import Model
from constants import IMAGE_SIZE, YOLO_MODEL_PATH

def filter_by_bboxes(bbox: Tuple[int, int, int, int]) -> bool:
    """
    Args:
        bbox: is the bounding box of a prediction, in xyxy format
                This means the shape of bbox is (leftmost x pixel, topmost y, rightmost x, bottommost y)
    """
    # TODO: Like in the other cases, return False if the bbox should not be considered.
    return True
def filter_by_scores(score: float) -> bool:
    """
    Args:
        score: the confidence score of a prediction
    """
    # Right now, this returns True for every object's confidence
    # TODO: Change this to filter the scores, or not at all
    # (returning True for all of them might be the right thing to do!)
    return True
def NUMBER_FRAMES_SKIPPED() -> int:
    # TODO: change this number to drop more frames
    # (must be a positive integer)
    return 0
def filter_by_classes(pred_class: int) -> bool:
    """
    Remember the class IDs:

        | Object    | ID    |
        | ---       | ---   |
        | Duckie    | 0     |
        | Cone      | 1     |
        | Truck     | 2     |
        | Bus       | 3     |


    Args:
        pred_class: the class of a prediction
    """
    # Right now, this returns True for every object's class
    # TODO: Change this to only return True for duckies!
    # In other words, returning False means that this prediction is ignored.
    return pred_class == 0

def compressed_imgmsg_to_cv2(msg):
    try:
        arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn("cv2.imdecode returned None! The image data may be corrupted.")
        return image
    except Exception as e:
        rospy.logerr(f"Failed to decode image: {e}")
        return None

def cv2_to_compressed_imgmsg(cv_image, stamp=None, frame_id=None, dst_format='jpeg'):
    """
    将 OpenCV (numpy) 图像编码为 ROS 的 CompressedImage 消息，不依赖 cv_bridge。
    :param cv_image: numpy 格式 BGR 图像
    :param stamp: 消息时间戳 (rospy.Time)
    :param frame_id: frame_id 字符串
    :param dst_format: 'jpeg' 或 'png'
    :return: sensor_msgs.msg.CompressedImage
    """
    msg = CompressedImage()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    if frame_id is not None:
        msg.header.frame_id = frame_id
    ext = '.jpg' if dst_format == 'jpeg' else '.png'
    success, encoded_img = cv2.imencode(ext, cv_image)
    if not success:
        rospy.logerr("cv2.imencode failed!")
        return None
    msg.format = dst_format
    msg.data = encoded_img.tobytes()
    return msg

class ObjectDetectionNode(DTROS):
    def __init__(self, node_name):
        # Initialize the DTROS parent class
        super(ObjectDetectionNode, self).__init__(node_name=node_name, node_type=NodeType.PERCEPTION)
        self.initialized = False
        self.log("Initializing!")
        self.veh = os.environ['VEHICLE_NAME']
        self.frame_id = 0
        self.model_wrapper = Model(YOLO_MODEL_PATH)
        #publish the image that after detection
        self.pub_detections_image = rospy.Publisher(
            "~image/compressed",
            CompressedImage,
            queue_size=1,
            dt_topic_type=TopicType.DEBUG
        )
        # Construct subscribers
        self.sub_image = rospy.Subscriber(
            f"/{self.veh}/camera_node/image/compressed",
            CompressedImage,
            self.image_cb,
            buff_size=10000000,
            queue_size=1,
        )
        self.log(f"debug= {rospy.get_param('~debug', False)}")
        self.log("Starting model loading!")
        self._debug = True
        self.log("Finished model loading!")
        self.first_image_received = False
        self.initialized = True
        self.log("Initialized!")

    def image_cb(self, image_msg):
        self.frame_id += 1
        self.frame_id = self.frame_id % (1 + NUMBER_FRAMES_SKIPPED())

        # Decode from compressed image with OpenCV
        try:
            bgr = compressed_imgmsg_to_cv2(image_msg)
        except ValueError as e:
            self.logerr("Could not decode image: %s" % e)
            return

        rgb = bgr[..., ::-1]

        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE))
        bboxes, classes, scores = self.model_wrapper.infer(rgb)

        detection = self.det2bool(bboxes, classes, scores)

        if self._debug:
            colors = {
                0: (0, 255, 255),  # cyan
                1: (0, 165, 255),  # orange
                2: (0, 250, 0),    # green
                3: (0, 0, 255),    # red
                4: (255, 0, 0),    # blue
                5: (128, 0, 128)   # purple
            }
            names = {
                0: "3_way_sign",
                1: "duckie",
                2: "duckiebot",
                3: "green_traffic_light",
                4: "red_traffic_light",
                5: "stop_sign"
            }
            font = cv2.FONT_HERSHEY_SIMPLEX
            for clas, box in zip(classes, bboxes):
                c = int(clas)
                pt1 = tuple(map(int, box[:2]))
                pt2 = tuple(map(int, box[2:]))
                color = tuple(reversed(colors.get(c, (255, 0, 0))))
                name = names.get(c, f"unknown_{c}")
                # draw bounding box
                rgb = cv2.rectangle(rgb, pt1, pt2, color, 2)
                # label location
                text_location = (pt1[0], min(pt2[1] + 30, IMAGE_SIZE))
                # draw label underneath the bounding box
                rgb = cv2.putText(rgb, name, text_location, font, 1, color, thickness=2)

            bgr = rgb[..., ::-1]
            obj_det_img = cv2_to_compressed_imgmsg(
                bgr,
                stamp=image_msg.header.stamp,  # 保留原始时间戳
                frame_id=image_msg.header.frame_id,  # 复制上游 frame_id（通常为 "camera"）
                dst_format='jpeg'
            )
            self.pub_detections_image.publish(obj_det_img)

    def det2bool(self, bboxes, classes, scores):
        box_ids = np.array(list(map(filter_by_bboxes, bboxes))).nonzero()[0]
        cla_ids = np.array(list(map(filter_by_classes, classes))).nonzero()[0]
        sco_ids = np.array(list(map(filter_by_scores, scores))).nonzero()[0]

        box_cla_ids = set(list(box_ids)).intersection(set(list(cla_ids)))
        box_cla_sco_ids = set(list(sco_ids)).intersection(set(list(box_cla_ids)))

        if len(box_cla_sco_ids) > 0:
            return True
        else:
            return False

if __name__ == "__main__":
    # Initialize the node
    object_detection_node = ObjectDetectionNode(node_name="object_detection_node")
    # Keep it spinning
    rospy.spin()
