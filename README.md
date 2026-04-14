# ONNX Heading Control

This repository is now set up to deploy the exported lane-following ONNX model
`rllib_db21j_multi_engine_602112_heading.onnx` on a Duckiebot.

The runtime pipeline is:

`camera -> crop/resize -> 3-frame stack -> ONNX heading -> heading_smooth -> wheel commands`

## Assets

The deployment assets are stored in `/assets`:

- `rllib_db21j_multi_engine_602112_heading.onnx`
- `rllib_db21j_multi_engine_602112_heading.onnx.json`

The node reads the metadata JSON so the preprocess settings stay aligned with training.

## Main node

The control node is:

- `packages/my_package/src/heading_control_node.py`

It subscribes to:

- `/<VEHICLE_NAME>/camera_node/image/compressed`

It publishes:

- `/<VEHICLE_NAME>/wheels_driver_node/wheels_cmd`
- `~heading`
- `~image/compressed` for debug visualization

## Build and run

Build the container:

```bash
dts devel build -f -H <ROBOT_NAME>
```

Run the default launcher:

```bash
dts devel run -H <ROBOT_NAME>
```

Or run the explicit heading-control launcher:

```bash
dts devel run -H <ROBOT_NAME> -L heading-control
```

## Useful ROS params

- `~model_path`
- `~model_metadata_path`
- `~forward_speed`
- `~max_steer`
- `~heading_type`
- `~publish_debug_image`
- `~process_every_n_frames`
- `~command_timeout`
- `~camera_topic`
- `~wheels_topic`

## Notes

- The ONNX model expects `84x84x9` NHWC input with pixel values normalized to `[0, 1]`.
- The current export is deterministic and outputs the heading mean used by `explore=False`.
- The wheel mapping matches the training wrapper:
  - `heading_smooth`: `heading = action^3 * max_steer`
  - `left = clip(1 + heading, 0, 1)`
  - `right = clip(1 - heading, 0, 1)`
