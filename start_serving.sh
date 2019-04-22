#!/usr/bin/env bash

PWD=`pwd`
docker run -it \
    -v $PWD:/work:ro \
    -w /work \
    -p 9001:9001 \
    kuberlab/serving:latest-openvino \
        kserving --driver openvino \
        --model-path models/facenet_pretrained_openvino_cpu/facenet.xml \
        --hooks serving_hook.py -o device=CPU \
        -o classifiers_dir=data/classifiers \
        -o bg_remove_dir=models/bg_remove \
        -o debug=true \
        --http-enable
