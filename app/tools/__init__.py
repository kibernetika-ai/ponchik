import sys

from app.recognize import defaults
from app.tools import images


def print_fun(s):
    print(s)
    sys.stdout.flush()


def add_normalization_args(parser):
    parser.add_argument(
        '--normalization',
        help='Image normalization during training.',
        default=defaults.NORMALIZATION,
        choices=[images.NORMALIZATION_PREWHITEN, images.NORMALIZATION_STANDARD, images.NORMALIZATION_FIXED],
    )

