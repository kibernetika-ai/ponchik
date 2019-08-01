from app.tools import images

ALIGNED_DIR = "./data/aligned"
CLASSIFIERS_DIR = "./data/classifiers"
NOT_DETECTED_DIR = "./data/notdetected"
NOT_DETECTED_CHECK_PERIOD = 3
MIN_FACE_SIZE = 30
IMAGE_SIZE = 160
IMAGE_MARGIN = 32
# MODEL_PATH = "./models/facenet_pretrained_openvino_cpu/facenet.xml"
MODEL_PATH = "./models/facenet_pretrained_vgg_openvino_cpu_1_0_0/facenet.xml"
AUG_FLIP = True
AUG_NOISE = 1
AUG_UPSCALE = True
NORMALIZATION = images.NORMALIZATION_FIXED
BATCH_SIZE = 10
DEVICE = "CPU"
CAMERA_DEVICE = "CV"
THRESHOLD = .5
HEAD_POSE_THRESHOLDS = [37., 35., 25.]
# FACE_DETECTION_PATH = (
#     '/opt/intel/openvino/deployment_tools/intel_models/'
#     'face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
# )
FACE_DETECTION_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'face-detection-adas-0001/FP32/face-detection-adas-0001.xml'
)
HEAD_POSE_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'head-pose-estimation-adas-0001/FP32/head-pose-estimation-adas-0001.xml'
)
PERSON_DETECTION_DRIVER = "openvino"
PERSON_DETECTION_PATH = (
    '/opt/intel/openvino/deployment_tools/intel_models/'
    'person-detection-retail-0013/FP32/person-detection-retail-0013.xml'
)
PERSON_THRESHOLD = .5
DEBUG = False
