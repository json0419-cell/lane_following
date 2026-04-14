#!/usr/bin/env python3

import os
import rospy
import numpy as np
from duckietown.dtros import DTROS, NodeType
from sensor_msgs.msg import CompressedImage, Image
import cv2

def compressed_image_to_cv2(msg):
    try:
        arr = np.frombuffer(msg.data, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            rospy.logwarn("cv2.imdecode returned None! The image data may be corrupted.")
        return image
    except Exception as e:
        rospy.logerr(f"Failed to decode image: {e}")
        return None

def cv2_to_imgmsg(cv_image, stamp=None, frame_id=None):
    # 用OpenCV和numpy构造ROS Image消息，不用cv_bridge
    msg = Image()
    msg.header.stamp = stamp if stamp is not None else rospy.Time.now()
    if frame_id is not None:
        msg.header.frame_id = frame_id
    msg.height = cv_image.shape[0]
    msg.width = cv_image.shape[1]
    msg.encoding = "bgr8"
    msg.is_bigendian = False
    msg.step = cv_image.shape[1] * 3
    msg.data = np.array(cv_image).tobytes()
    return msg

class CameraRelayNode(DTROS):
    def __init__(self, node_name):
        super(CameraRelayNode, self).__init__(
            node_name=node_name,
            node_type=NodeType.VISUALIZATION
        )
        self._vehicle_name = os.environ.get('VEHICLE_NAME', 'duckiebot')
        self._camera_topic = f"/{self._vehicle_name}/camera_node/image/compressed"
        self._pub_topic = f"/{self._vehicle_name}/camera_reader_node/image_raw"
        self.pub = rospy.Publisher(self._pub_topic, Image, queue_size=1)
        self.sub = rospy.Subscriber(self._camera_topic, CompressedImage, self.callback)
        rospy.loginfo(f"[{node_name}] Subscribing to {self._camera_topic}, publishing to {self._pub_topic}")

    def callback(self, msg):
        image = compressed_image_to_cv2(msg)
        if image is not None:
            img_msg = cv2_to_imgmsg(image, stamp=msg.header.stamp, frame_id=msg.header.frame_id)
            self.pub.publish(img_msg)
            rospy.loginfo_throttle(2, f"[{self._vehicle_name}] Published Image. shape: {image.shape}")
        else:
            rospy.logwarn(f"[{self._vehicle_name}] Failed to decode image!")

if __name__ == '__main__':
    node = CameraRelayNode(node_name='camera_reader_node')
    rospy.spin()
