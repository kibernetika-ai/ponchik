import argparse
import os

from svod_rcgn.recognize import defaults
from svod_rcgn.tools import downloader

parser = argparse.ArgumentParser()
parser.add_argument(
    '--bg_remove_url',
    type=str,
    default="https://dev.kibernetika.io/api/v0.2/workspace/kuberlab-demo/mlmodel/coco-bg-rm/versions/"
            "1.0.0/download/model-coco-bg-rm-1.0.0.tar",
    help='URL to background remove model',
)
parser.add_argument(
    '--facenet_pretrained_openvino_cpu_url',
    type=str,
    default="https://dev.kibernetika.io/api/v0.2/workspace/kuberlab-demo/mlmodel/facenet-pretrained/versions/"
            "1.0.0-openvino-cpu/download/model-facenet-pretrained-1.0.0-openvino-cpu.tar",
    help='URL to background remove model',
)
args = parser.parse_args()
err = downloader.Downloader(args.bg_remove_url, './models/bg_remove').extract()
if err is not None:
    raise ValueError(err)
err = downloader.Downloader(args.facenet_pretrained_openvino_cpu_url, os.path.dirname(defaults.MODEL_PATH)).extract()
if err is not None:
    raise ValueError(err)
