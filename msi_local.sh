#!/usr/bin/env bash
source activate svod

source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

./streaming.sh \
    --token '469d7155-c74f-4c88-aed9-e5ca9c8b3411' \
    --input realsense \
    --output hls \
    --base-url https://dev.kibernetika.io/api/v0.2 \
    --model-name ponchik \
    --workspace-name expo-recall \
    --inference-fps 0 \
    --head-pose-path models/head_pose/head-pose-estimation-adas-0001.xml \
    --slack-token '' \
    --slack-channel ''
