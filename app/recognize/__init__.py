from app.recognize import defaults


def add_common_args(parser):
    parser.add_argument(
        '--face_detection_path',
        default=defaults.FACE_DETECTION_PATH,
        help='Path to face-detection openvino model',
    )
    parser.add_argument(
        '--head_pose_path',
        default=defaults.HEAD_POSE_PATH,
        help='Path to head-pose-estimation openvino model',
    )
    parser.add_argument(
        '--head_pose_not_account',
        default=False,
        action='store_true',
        help='Do not take into account head-pose result. Just process and give output',
    )
    parser.add_argument(
        '--classifiers_dir',
        help='Path to classifier models stored as pickle (.pkl) files.',
        type=str,
        default=defaults.CLASSIFIERS_DIR,
    )
    parser.add_argument(
        '--model_path',
        type=str,
        help='OpenVINO IR model xml path',
        default=defaults.MODEL_PATH,
    )
    parser.add_argument(
        '--without_facenet',
        help='Do not load OpenVINO IR model (process without calculating embeddings)',
        action='store_true',
    )
    parser.add_argument(
        '--without_classifiers',
        help='Do not load classifiers (process without face recognition)',
        action='store_true',
    )
    parser.add_argument(
        '--min_face_size',
        type=int,
        help='Minimum face size in pixels.',
        default=defaults.MIN_FACE_SIZE,
    )
    parser.add_argument(
        '--multi_detect',
        help='Multi detection steps, comma separated. Example: 3,4',
    )
    parser.add_argument(
        '--device',
        help='Device for openVINO.',
        default=defaults.DEVICE,
        choices=[defaults.DEVICE, "MYRIAD"],
    )
    parser.add_argument(
        '--process_not_detected',
        help='Post process not detected faces.',
        action='store_true',
    )
