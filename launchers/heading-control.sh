#!/bin/bash
source /environment.sh

dt-launchfile-init
dt-exec rosrun my_package heading_control_node.py
dt-launchfile-join
