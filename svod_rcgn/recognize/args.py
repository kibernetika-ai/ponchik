from svod_rcgn.recognize import defaults


def add_common_args(parser):
    parser.add_argument(
        '--face_detection_path',
        default=defaults.FACE_DETECTION_PATH,
        help='Path to face-detection-retail openvino model',
    )
    parser.add_argument(
        '--classifiers_dir',
        help='Path to classifier models stored as pickle (.pkl) files.',
        type=str,
        default=defaults.CLASSIFIERS_DIR,
    )
    parser.add_argument(
        '--model_dir',
        type=str,
        help='OpenVINO IR model directory',
        default=defaults.MODEL_DIR,
    )
    parser.add_argument(
        '--device',
        help='Device for openVINO.',
        default=defaults.DEVICE,
        choices=["CPU", "MYRIAD"],
    )


