#!/usr/bin/env bash

CURRENT_DIR=`pwd`
docker run -it \
    -v $CURRENT_DIR:/work:ro \
    -v $CURRENT_DIR/data/clarified:/clarified \
    -v $CURRENT_DIR/data/uploaded:/uploaded \
    -w /work \
    -p 9001:9001 \
    kuberlab/serving:latest-openvino \
        kserving --driver openvino \
        --model-path models/facenet_pretrained_openvino_cpu/facenet.xml \
        --hooks serving_hook.py -o device=CPU \
        -o classifiers_dir=data/classifiers \
        -o clarified_dir=/clarified \
        -o uploaded_dir=/uploaded \
        -o debug=true \
        --http-enable
