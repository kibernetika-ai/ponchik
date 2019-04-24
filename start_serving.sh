#!/usr/bin/env bash

CURRENT_DIR=`pwd`
docker run -it \
    -v $CURRENT_DIR:/work:ro \
    -w /work \
    -p 9001:9001 \
    kuberlab/serving:latest-openvino \
        kserving --driver openvino \
        --model-path models/facenet_pretrained_openvino_cpu/facenet.xml \
        --hooks serving_hook.py -o device=CPU \
        -o classifiers_dir=data/classifiers \
        -o debug=true \
        --http-enable
