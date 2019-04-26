# svod-rcgn

## Default project data and models directories structure

By default all app data is in the next directories:
```
.
├── data
│   ├── aligned
│   ├── clarify
│   ├── classifiers
│   └── faces
└── models
    ├── bg_remove
    └── facenet_pretrained_openvino_cpu
```
Each directory can be redefined by command line arguments.

| Directory | CLI argument to redefine | Description |
|-----------|--------------------------|-------------|
| `./data/faces` | `--input_dir` | Initial dataset with images in named directories for each man. Initial faces is available in [this dataset](https://dev.kibernetika.io/svod/catalog/dataset/svod-faces/versions/1.0.0). |
| `./data/aligned` | `--aligned_dir` | Prepared for training dataset's files |
| `./data/classifiers` | `--classifiers_dir` | Directory for trained classifiers `classifier-*.pkl` |
| `./models/bg_remove` | `--bg_remove_path` | Directory for background remove tf model. |
| `./models/facenet_pretrained_openvino_cpu` | `--model` | Directory fot OpenVINO IR model. |

## Endpoints

### Download required models

```bash
python download_models.py
```

Download models to default model directories:
* [coco-bg-rm:1.0.0](https://dev.kibernetika.io/kuberlab-demo/catalog/mlmodel/coco-bg-rm/versions/1.0.0) to `./models/bg_remove`
* [facenet-pretrained:1.0.0-openvino-cpu](https://dev.kibernetika.io/kuberlab-demo/catalog/mlmodel/facenet-pretrained/versions/1.0.0-openvino-cpu) to `.models/facenet_pretrained_openvino_cpu`

### Fill meta to dataset

Fill scraped meta data info from [scrape-linkedin-profiles](https://github.com/monstarnn/scrape-linkedin-profiles) to dataset: 

```bash
python fill_meta.py ./data/faces data.linkedin.scraped.csv --column_data name=0 position=1 company=2 linkedin=3 positions=5 companies=6 links=7
```

### Prepare and train

Prepare dataset (align images, calculate embeddings) and train SMV and kNN classifiers:

```bash
python prepare.py
```
Argument `--download` can be used for downloading dataset's archive from specified URL.
This argument can be used with `--clear_input_dir` to clear source data dir before archive downloading. For example:
```bash
python prepare.py --download https://dev.kibernetika.io/api/v0.2/workspace/kuberlab-demo/dataset/faces-svod/versions/0.0.1/download/dataset-faces-svod-0.0.1.tar --clear_input_dir 
```
Argument `--complementary_align` allows to **add** new classes and images to existing alignment data.
Argument `--complementary_train` uses existing embeddings for classifiers training.
So adding some new persons to classifiers from another input directory looks like this: 
```bash
python prepare.py --input_dir ./data/another_input_dir --complementary_align --complementary_train
```
Argument `--clarified` says that alignment input data is from servings images that was clarified, so this images will not detect for faces and uses as is and alignment will be complementary (key `--complementary_align` is not needed):
```bash
python prepare.py --input_dir ./data/clarify --clarified
```

It's available to skip alignment `--skip_alignment` or skip training `--skip_training`.

If recognizing process is running, this script will reload classifiers for running process (if `--skip_training` is not specified).

### Camera and recognizing

Open window with camera's video stream and draw labeled boxes for recognized faces: 

```bash
python video.py
```

Default video source is main web-camera, but it's can be redefined with argument `--video_source`.
Use argument `--video_async` for asynchronous video rendering (by default **each video's frame** waits for faces recognition, detection and add boxes and labels to this frame).

## Listener interface

It's available to do any actions during face recognition process.
Start faces recognition:
```bash
python video.py
```

In this time it's available to make calls:
```python
from svod_rcgn.control.client import SVODClient

cl = SVODClient()
cl.call(action, data)
```

Available actions:
* `cl.call("reload_classifiers")` - for classifiers reloading
* `cl.call("debug", True|False)` - for enabling/disabling debug output
