#!/usr/bin/env bash

clf_path=data/classifiers
inference_fps=10
face_detection_path=/opt/intel/openvino/deployment_tools/intel_models/face-detection-retail-0004/FP32/face-detection-retail-0004.xml
model_path=models/facenet_pretrained_openvino_cpu/facenet.xml
head_pose_path=models/head_pose/head-pose-estimation-adas-0001.xml
backend=srs # livego / rtmp-mux/ srs
OUTPUT=""
INPUT="server"
rs_file=""
token=""
base_url="https://dev.kibernetika.io/api/v0.2"

slack_token=""
slack_channel=""


usage="""
Usage:
  $0 <options>

Options:
  
  --output <display/rtmp-url> Output type / RTMP URL stream address. Optional.
  --classifiers <dir> Classifiers directory, default $clf_path.
  --inference-fps <int> Inference FPS, default $inference_fps.
  --model-path <path> Facenet model path, default $model_path
  --head-pose-path <path> Path to head-pose-model.
  --text-model-path <path> Text detection model path
  --ocr-model-path <path> OCR model path
  --input <type> Input type. Can be one of 'camera', 'realsense', 'server'
                 or any opencv-compatible URL (rtmp/rtsp/filepath etc.). Default $INPUT.
  --input-realsense Path to realsense .bag file to stream.
  
  --token Token for authentication in Kibernetika.AI
  --base-url Base URL of Kibernetika.AI API. Default $base_url.
  
  --slack-token   Slack token for sending notifications to Slack.
  --slack-channel Slack channel name for notifications.


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
    --text-model-path)
    text_model_path="$2"
    shift; shift
    ;;
    --ocr-model-path)
    ocr_model_path="$2"
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
    --slack-token)
    slack_token="$2"
    shift; shift;
    ;;
    --slack-channel)
    slack_channel="$2"
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

pull_model_args=""
if [ ! -z "$token" ] && [ ! -z "$base_url" ];
then
  echo "Enable pull model"
  pull_model_args="-o enable_pull_model=true -o base_url=$base_url -o token=$token"
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


kstreaming --driver openvino --model-path $face_detection_path --driver openvino --model-path $model_path \
  --driver tensorflow --model-path=$text_model_path \
  --driver tensorflow --model-path=$ocr_model_path $head_pose_args --hooks serving_hook.py -o classifiers_dir=$clf_path \
  -o need_table=false -o timing=false -o output_type=image --input $INPUT $output_arg --rs-file "$rs_file" \
   --initial-stream live --input-name input --output-name output --rtmp-backend $backend \
  -o enable_log=true -o inference_fps=$inference_fps $pull_model_args -o slack_token="$slack_token" \
  -o slack_channel="$slack_channel" -o badge_detector=yes -o skip_frames=true -o min_face_size=50

