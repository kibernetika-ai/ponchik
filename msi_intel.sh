#!/usr/bin/env bash
source activate svod

source /opt/intel/openvino/bin/setupvars.sh -pyver 3.6

./streaming.sh \
    --token '37d76289-6b01-44a7-b438-614fd910c047' \
    --input realsense \
    --output hls \
    --base-url https://cloud.kibernetika.io/api/v0.2 \
    --inference-fps 4 \
    --head-pose-path models/head_pose/head-pose-estimation-adas-0001.xml \
    --slack-token '' \
    --slack-channel ''
