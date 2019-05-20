from app.dataset import aligner
from app.dataset.aligner import Aligner


def add_aligner_args(parser):
    parser.add_argument(
        '--input_dir',
        type=str,
        help='Directory with source images.',
        default=aligner.DEFAULT_INPUT_DIR,
    )
    parser.add_argument(
        '--clarified',
        help='Source is clarified data dir. Sets alignment as complementary (see --complementary_align)',
        action="store_true",
    )
    parser.add_argument(
        '--clear_input_dir',
        help='Clear input dir before extracting downloaded archive.',
        action="store_true",
    )
    parser.add_argument(
        '--complementary_align',
        help='Existing aligned images in aligned dir supplements with new ones from input dir.',
        action='store_true',
    )
    parser.add_argument(
        '--download',
        type=str,
        help='URL to .tar or .tar.gz dataset with source faces.',
        default=None,
    )


def aligner_args(args):
    return Aligner(
        input_dir=args.input_dir,
        clarified=args.clarified,
        clear_input_dir=args.clear_input_dir,
        download=args.download,
        aligned_dir=args.aligned_dir,
        complementary_align=args.complementary_align or args.complementary,
        min_face_size=args.min_face_size,
        image_size=args.image_size,
        margin=args.margin,
        face_detection_path=args.face_detection_path,
        bg_remove_path=args.bg_remove_path,
        device=args.device,
    )
