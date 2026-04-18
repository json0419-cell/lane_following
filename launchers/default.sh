#!/bin/bash

source /environment.sh

# initialize launch file
dt-launchfile-init

# YOUR CODE BELOW THIS LINE
# ----------------------------------------------------------------------------


# NOTE: Use the variable DT_REPO_PATH to know the absolute path to your code
# NOTE: Use `dt-exec COMMAND` to run the main process (blocking process)

# launching app
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


# ----------------------------------------------------------------------------
# YOUR CODE ABOVE THIS LINE

# wait for app to end
dt-launchfile-join
