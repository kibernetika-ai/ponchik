FROM kuberlab/serving:latest-openvino

RUN pip install croniter slackclient srt

RUN mkdir /svod

COPY ./models /svod/models
COPY ./data /svod/data
COPY ./app /svod/app
COPY ./streaming.sh /svod/streaming.sh
COPY ./serving_hook.py /svod/serving_hook.py
COPY ./pull_model.py /svod/pull_model.py
COPY ./get_stats.py /svod/get_stats.py

RUN mkdir -p /svod/data/classifiers
RUN mkdir -p /svod/log