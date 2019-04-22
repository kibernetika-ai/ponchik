ALIGNED_DIR = "./data/aligned"
CLASSIFIERS_DIR = "./data/classifiers"
MIN_FACE_SIZE = 25
IMAGE_SIZE = 160
IMAGE_MARGIN = 32
MODEL_DIR = "./models/facenet_pretrained_openvino_cpu"
AUG_FLIP = True
AUG_NOISE = 3
BATCH_SIZE = 10
DEVICE = "CPU"
CAMERA_DEVICE = "CV"
THRESHOLD = .5
FACE_DETECTION_PATH = '/opt/intel/openvino/deployment_tools/intel_models/' \
                      'face-detection-retail-0004/FP32/face-detection-retail-0004.xml'
DEBUG = False
