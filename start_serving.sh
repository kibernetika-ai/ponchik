#!/usr/bin/env bash

CURRENT_DIR=`pwd`
docker run -it \
    -v $CURRENT_DIR:/work:ro \
    -v $CURRENT_DIR/data/clarify_data:/clarify_data \
    -w /work \
    -p 9001:9001 \
    kuberlab/serving:latest-openvino \
        kserving --driver openvino \
        --model-path models/facenet_pretrained_openvino_cpu/facenet.xml \
        --hooks serving_hook.py -o device=CPU \
        -o classifiers_dir=data/classifiers \
        -o clarify_dir=/clarify_data \
        -o debug=true \
        --http-enable
