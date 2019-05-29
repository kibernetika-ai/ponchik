ALIGNED_DIR = "./data/aligned"
CLASSIFIERS_DIR = "./data/classifiers"
NOT_DETECTED_DIR = "./data/notdetected"
NOT_DETECTED_CHECK_PERIOD = 3
MIN_FACE_SIZE = 30
IMAGE_SIZE = 160
IMAGE_MARGIN = 32
MODEL_PATH = "./models/facenet_pretrained_openvino_cpu/facenet.xml"
AUG_FLIP = True
AUG_NOISE = 3
BATCH_SIZE = 10
DEVICE = "CPU"
CAMERA_DEVICE = "CV"
THRESHOLD = .5
HEAD_POSE_THRESHOLDS = [37., 35., 25.]
FACE_DETECTION_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
)
HEAD_POSE_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml'
)
DEBUG = False
