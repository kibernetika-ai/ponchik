rtmp_url=$1
clf_path=$2
inference_fps=$3
model_path=$4
backend=livego # livego / rtmp-mux

if [ -z "$rtmp_url" ];
then
  echo "Usage: $0 <output_rtmp_url> [classifiers-dir] [inference_fps; default=2] [facenet-model-path; default=models/facenet_pretrained_openvino_cpu/facenet.xml]"
  echo
  echo "Then, use ffmpeg to stream to this serving, e.g:"
  echo 
  echo "  ffmpeg -re -i <video/source> -vcodec libx264 -acodec aac -f flv rtmp://localhost/live"
  echo 
  exit 1
fi

if [ -z "$clf_path" ];
then
  clf_path=data/classifiers
fi

if [ -z "$inference_fps" ];
then
  inference_fps=2
fi

if [ -z "$model_path" ];
then
  model_path=models/facenet_pretrained_openvino_cpu/facenet.xml
fi

kstreaming --driver openvino --model-path $model_path --hooks serving_hook.py -o classifiers_dir=$clf_path -o need_table=false -o timing=false -o output_type=image --input server --output-type rtmp --output-rtmp $rtmp_url --initial-stream live --input-name input --output-name output --rtmp-backend $backend -o enable_log=true -o inference_fps=$inference_fps

#kstreaming --driver openvino --model-path ~/projects/kiber-facenet/src/train-new/facenet.xml --hooks serving_hook.py -o classifiers_dir=clf -o need_table=false -o timing=false -o output_type=image --input server --output-type display --initial-stream live --input-name input --output-name output --rtmp-backend livego -o enable_log=true
