#!/bin/bash
source /environment.sh

dt-launchfile-init
ROS_ARGS=()

if [[ "${HEADING_RECORD_DEBUG:-0}" == "1" ]]; then
    ROS_ARGS+=("_record_debug:=true")
fi

if [[ -n "${HEADING_RECORD_DIR:-}" ]]; then
    ROS_ARGS+=("_record_debug_dir:=${HEADING_RECORD_DIR}")
fi

if [[ -n "${HEADING_RECORD_MAX_FRAMES:-}" ]]; then
    ROS_ARGS+=("_record_max_frames:=${HEADING_RECORD_MAX_FRAMES}")
fi

dt-exec rosrun my_package heading_control_node.py "${ROS_ARGS[@]}"
dt-launchfile-join
