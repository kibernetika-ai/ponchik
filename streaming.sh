rtmp_url=""
clf_path=data/classifiers
inference_fps=10
model_path=models/facenet_pretrained_openvino_cpu/facenet.xml
backend=livego # livego / rtmp-mux
OUTPUT_TYPE="rtmp"
INPUT="server"
rs_file=""

usage="""
Usage:
  $0 <options>

Options:
  
  --output-type <rtmp/display> Output type: rtmp/display/'', default $OUTPUT_TYPE.
  --classifiers <dir> Classifiers directory, default $clf_path.
  --inference-fps <int> Inference FPS, default $inference_fps.
  --model-path <path> Facenet model path, default $model_path
  --output-rtmp <rtmp-url> Output RTMP URL stream address. Optional.
  --input <type> Input type. Can be one of 'camera', 'realsense', 'server' 
                 or any opencv-compatible URL (rtmp/rtsp/filepath etc.). Default $INPUT.
  --input-realsense Path to realsense .bag file to stream.


In case of --output-type rtmp use ffmpeg to stream to this serving, e.g:
 
  ffmpeg -re -i <video/source> -vcodec libx264 -acodec aac -f flv rtmp://localhost/live
"""

while [[ $# -gt 0 ]]
do
key="$1"
  case $key in
    --output-type)
    OUTPUT_TYPE="$2"
    shift # past argument
    shift # past value
    ;;
    --classifiers)
    clf_path="$2"
    shift; shift
    ;;
    --inference-fps)
    inference_fps="$2"
    shift; shift
    ;;
    --model-path)
    model_path="$2"
    shift; shift
    ;;
    --output-rtmp)
    rtmp_url="$2"
    shift; shift;
    ;;
    --input)
    INPUT="$2"
    shift; shift
    ;;
    --input-realsense)
    rs_file="$2"
    shift; shift;
    ;;
    *)
    echo "Unknown option $key."
    echo "$usage"
    exit 1
    shift
  esac
done

#echo "INPUT=$INPUT"
#echo "OUTPUT_TYPE=$OUTPUT_TYPE"
#echo "clf_path=$clf_path"
#echo "inference_fps=$inference_fps"
#echo "model_path=$model_path"

if [ "$OUTPUT_TYPE" == "rtmp" ];
then
  if [ -z "$rtmp_url" ];
  then
    echo "--output-rtmp required in case of providing --output-type rtmp"
    echo
    echo "$usage"
    exit 1
  fi
fi

output_arg="--output-type $OUTPUT_TYPE"
if [ -z "$OUTPUT_TYPE" ];
then
  output_arg=""
fi

kstreaming --driver openvino --model-path $model_path --hooks serving_hook.py -o classifiers_dir=$clf_path -o need_table=false -o timing=false -o output_type=image --input $INPUT $output_arg --rs-file "$rs_file" --output-rtmp "$rtmp_url" --initial-stream live --input-name input --output-name output --rtmp-backend $backend -o enable_log=true -o inference_fps=$inference_fps

