import argparse

from svod_rcgn.mlboard import mlboard, update_task_info, catalog_ref
from svod_rcgn.recognize.args import add_common_args
from svod_rcgn.tools.print import print_fun


def main():

    if not mlboard:
        print_fun("Skipped: no mlboard detected")
        return

    parser = argparse.ArgumentParser()
    add_common_args(parser)
    parser.add_argument(
        'model_name',
        help='Exported model name.',
        type=str,
    )
    parser.add_argument(
        'model_version',
        help='Exported model version.',
        type=str,
    )
    args = parser.parse_args()

    print_fun('Uploading model...')
    mlboard.model_upload(args.model_name, args.model_version, args.classifiers_dir)
    update_task_info({'model_reference': catalog_ref(args.model_name, 'mlmodel', args.model_version)})
    print_fun("New model uploaded as '%s', version '%s'." % (args.model_name, args.model_version))


if __name__ == '__main__':
    main()