import argparse

from prometheus_client import parser
import requests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--endpoint',
        default='http://localhost:9090/metrics'
    )

    return parser.parse_args()


def main():
    args = parse_args()
    metrics_raw = requests.get(args.endpoint).content.decode()
    metrics = parser.text_string_to_metric_families(metrics_raw)
    metrics_dict = {}
    for family in metrics:
        if family.name in metrics_dict:
            metrics_dict[family.name].extend(family.samples)
        else:
            metrics_dict[family.name] = family.samples

    print('Inference average time: {:.3f}ms'.format(metrics_dict['serving_request_ms_avg_time'][0].value))
    rtime = metrics_dict['serving_request_time'][0].value
    fps = 1 / rtime * 1000 if rtime != 0 else 0
    print('Instant inference FPS: {}'.format(fps))


if __name__ == '__main__':
    main()
