import argparse

from svod_rcgn.dataset.aligner import add_aligner_args, aligner_args
from svod_rcgn.recognize.args import add_common_args
from svod_rcgn.recognize.classifiers import add_classifier_args, classifiers_args
from svod_rcgn.tools.bg_remove import add_bg_remove_args


def main():
    parser = argparse.ArgumentParser()
    add_aligner_args(parser)
    add_common_args(parser)
    add_classifier_args(parser)
    add_bg_remove_args(parser)
    parser.add_argument(
        '--skip_align',
        help='Skip alignment for source images from input dir, only calculate embeddings and train classifiers.',
        action='store_true',
    )
    parser.add_argument(
        '--skip_train',
        help='Skip calculating embeddings and training, only alignment for source images from input dir.',
        action='store_true',
    )
    args = parser.parse_args()
    if not args.skip_align:
        al = aligner_args(args)
        al.align()
    if not args.skip_train:
        clf = classifiers_args(args)
        clf.train()
        from svod_rcgn.control.client import SVODClient
        cl = SVODClient()
        cl.call('reload_classifiers')


if __name__ == '__main__':
    main()
