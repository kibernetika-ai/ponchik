#!/usr/bin/env bash

set -x

clf_path=data/classifiers
inference_fps=10
#face_detection_path=/opt/intel/openvino/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
face_detection_path=/opt/intel/openvino/deployment_tools/intel_models/face-detection-adas-0001/FP32/face-detection-adas-0001.xml
model_path=models/facenet_pretrained_openvino_cpu/facenet.xml
head_pose_path=models/head_pose/head-pose-estimation-adas-0001.xml
backend=srs # livego / rtmp-mux / srs
OUTPUT=""
INPUT="server"
rs_file=""
token=""
multi_detect=""
workspace_name="intel"
model_name="faces"
base_url="https://dev.kibernetika.io/api/v0.2"

slack_token=""
slack_channel=""
slack_server=""

raw_args="-o min_face_size=40"

usage="""
Usage:
  $0 <options>

Options:
  
  --output <display/rtmp-url> Output type / RTMP URL stream address. Optional.
  --classifiers <dir> Classifiers directory, default $clf_path.
  --multi-detect <int>,<int>,... Multi detect steps, comma separated. Recommended value 2 or 3.
  --inference-fps <int> Inference FPS, default $inference_fps.
  --model-path <path> Facenet model path, default $model_path
  --head-pose-path <path> Path to head-pose-model
  --input <type> Input type. Can be one of 'camera', 'realsense', 'server' 
                 or any opencv-compatible URL (rtmp/rtsp/filepath etc.). Default $INPUT.
  --input-realsense Path to realsense .bag file to stream.
  
  --token Token for authentication in Kibernetika.AI
  --base-url Base URL of Kibernetika.AI API. Default $base_url.
  
  --slack-token   Slack token for sending notifications to Slack.
  --slack-channel Slack channel name for notifications.
  --slack-server Slack server address for notifications.

  --raw <args> Raw args for streaming. Example: '-o threshold=0.7 -o min_face_size=30'.


In case of --output-type rtmp use ffmpeg to stream to this serving, e.g:
 
  ffmpeg -re -i <video/source> -vcodec libx264 -acodec aac -f flv rtmp://localhost/live
"""

while [[ $# -gt 0 ]]
do
key="$1"
  case $key in
    --output)
    OUTPUT="$2"
    shift # past argument
    shift # past value
    ;;
    --classifiers)
    clf_path="$2"
    shift; shift
    ;;
    --multi-detect)
    multi_detect="$2"
    shift; shift
    ;;
    --raw)
    raw_args="$2"
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
    --head-pose-path)
    head_pose_path="$2"
    shift; shift
    ;;
    --input)
    INPUT="$2"
    shift; shift
    ;;
    --input-realsense)
    rs_file="$2"
    shift; shift;
    ;;
    --token)
    token="$2"
    shift; shift;
    ;;
    --base-url)
    base_url="$2"
    shift; shift;
    ;;
    --model-name)
    model_name="$2"
    shift; shift;
    ;;
    --workspace-name)
    workspace_name="$2"
    shift; shift;
    ;;
    --slack-token)
    slack_token="$2"
    shift; shift;
    ;;
    --slack-channel)
    slack_channel="$2"
    shift; shift;
    ;;
    --slack-server)
    slack_server="$2"
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

output_arg='--output '"$OUTPUT"''
if [ -z "$OUTPUT" ];
then
  output_arg=""
fi

multi_detect_args=""
if [ ! -z "$multi_detect" ];
then
  multi_detect_args="-o multi_detect=$multi_detect"
fi

pull_model_args=""
if [ ! -z "$token" ] && [ ! -z "$base_url" ];
then
  echo "Enable pull model"
  pull_model_args="-o enable_pull_model=true -o base_url=$base_url -o token=$token -o model_name=$model_name -o workspace_name=$workspace_name"
fi

head_pose_args=""
if [ ! -z "$head_pose_path" ];
then
  if [ -f $head_pose_path ];
  then
    head_pose_args="--driver openvino --model-path $head_pose_path"
  else
    echo "Head pose model not found in: $head_pose_path"
  fi
fi

kstreaming --driver openvino --model-path $face_detection_path --driver openvino --model-path $model_path $head_pose_args \
 --hooks serving_hook.py -o classifiers_dir=$clf_path \
 -o need_table=false -o timing=false -o output_type=image --input $INPUT $output_arg --rs-file "$rs_file" \
  --initial-stream live --input-name input --output-name output -o skip_frames=true \
 --rtmp-backend $backend -o enable_log=false -o only_distance=true -o debug=true -o inference_fps=0 $pull_model_args \
 -o slack_token="$slack_token" -o slack_channel="$slack_channel" -o slack_server="$slack_server" \
 $multi_detect_args $raw_args --http-enable --disable-predict-log --output_fps $inference_fps

