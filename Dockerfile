FROM kuberlab/serving:latest-openvino

RUN pip install croniter slackclient srt
COPY ./ /svod

RUN mkdir /svod/data