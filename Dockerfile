FROM kuberlab/serving:latest-openvino

RUN pip install croniter slackclient srt

RUN mkdir /svod

COPY ./models /svod/models
COPY ./app /svod/app
COPY ./streaming.sh /svod/streaming.sh
COPY ./serving_hook.py /svod/serving_hook.py
COPY ./pull_model.py /svod/pull_model.py

RUN mkdir -p /svod/data/classifiers