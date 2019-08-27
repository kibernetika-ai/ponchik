import logging
import sys

from app.recognize import defaults
from app.tools import images


def box_intersection(box_a, box_b):
    xa = max(box_a[0], box_b[0])
    ya = max(box_a[1], box_b[1])
    xb = min(box_a[2], box_b[2])
    yb = min(box_a[3], box_b[3])

    inter_area = max(0, xb - xa) * max(0, yb - ya)

    box_a_area = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    box_b_area = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])

    d = float(box_a_area + box_b_area - inter_area)
    if d == 0:
        return 0
    iou = inter_area / d
    return iou


def print_fun(s):
    print(s)
    sys.stdout.flush()


def add_normalization_args(parser):
    parser.add_argument(
        '--normalization',
        help='Image normalization during training.',
        default=defaults.NORMALIZATION,
        choices=[
            images.NORMALIZATION_PREWHITEN,
            images.NORMALIZATION_STANDARD,
            images.NORMALIZATION_FIXED,
            images.NORMALIZATION_NONE,
        ],
    )


def boolean_string(s):
    if isinstance(s, bool):
        return s

    s = s.lower()
    if s not in {'false', 'true'}:
        raise ValueError('Not a valid boolean string')
    return s == 'true'


def box_includes(outer, inner):
    return outer[0] <= inner[0] and \
           outer[1] <= inner[1] and \
           outer[2] >= inner[2] and \
           outer[3] >= inner[3]


def configure_logging():
    logging.basicConfig(
        format='%(asctime)s %(levelname)-5s %(name)-10s [-] %(message)s',
        level='INFO'
    )
    logging.root.setLevel(logging.INFO)
