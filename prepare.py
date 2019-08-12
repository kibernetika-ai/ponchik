import argparse

from app.dataset import add_aligner_args, aligner_args
from app.recognize import add_common_args
from app.recognize.classifiers import add_classifier_args, classifiers_args
from app.tools.bg_remove import add_bg_remove_args
from app.tools import print_fun, add_normalization_args


def main():
    parser = argparse.ArgumentParser()
    add_aligner_args(parser)
    add_common_args(parser)
    add_classifier_args(parser)
    add_bg_remove_args(parser)
    add_normalization_args(parser)
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
    parser.add_argument(
        '--align_images_limit',
        type=int,
        help='Set limit for processed images in alignment.',
        default=None,
    )
    parser.add_argument(
        '--model_name',
        help='Exported model name.',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--model_version',
        help='Exported model version.',
        type=str,
        default=None,
    )
    parser.add_argument(
        '--complementary',
        help='Complementary align and training.',
        action='store_true',
    )
    args = parser.parse_args()
    if not args.skip_align:
        al = aligner_args(args)
        al.align(args.align_images_limit)
    if not args.skip_train:
        clf = classifiers_args(args)
        clf.train()
        from app.control.client import Client
        cl = Client()
        cl.call('reload_classifiers')
        if args.model_name is not None and args.model_version is not None:
            from app.mlboard import mlboard, update_task_info, catalog_ref
            print_fun('Uploading model...')
            model_version = args.model_version
            if args.device not in args.model_version:
                model_version = args.model_version + '-%s' % args.device
            mlboard.model_upload(args.model_name, model_version, args.classifiers_dir)
            update_task_info({'model_reference': catalog_ref(args.model_name, 'mlmodel', args.model_version)})
            print_fun("New model uploaded as '%s', version '%s'." % (args.model_name, args.model_version))


if __name__ == '__main__':
    main()
