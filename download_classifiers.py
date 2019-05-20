import argparse

from app.recognize import add_common_args
from app.tools import downloader

parser = argparse.ArgumentParser()
add_common_args(parser)
parser.add_argument(
    'classifiers_url',
    type=str,
    help='URL for the pretrained classifiers',
)
args = parser.parse_args()

err = downloader.Downloader(args.classifiers_url, args.classifiers_dir).extract()
if err is not None:
    raise ValueError(err)
