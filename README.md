# svod-rcgn

## Default project data and models directories structure

By default all app data is in the next directories (with arguments to change default dirs):
```
.
├── data
│   ├── aligned                          --aligned_dir
│   ├── classifiers                      --classifiers_dir
│   └── faces                            --input_dir
└── models
    ├── bg_remove                        --bg_remove_path
    └── facenet_pretrained_openvino_cpu  --model
```
Each directory can be redefined by command line arguments.

| Directory | Description |
|-----------|-------------|
| `./data/faces` | Initial dataset with images in named directories for each man. Initial faces is available in [this dataset](https://dev.kibernetika.io/svod/catalog/dataset/svod-faces/versions/1.0.0). |
| `./data/aligned` | Prepared for training dataset's files |
| `./data/classifiers` | Directory for trained classifiers `classifier-*.pkl`, meta data `meta.json` and faces previews in directory `previews` |
| `./models/bg_remove` | Directory for background remove tf model. |
| `./models/facenet_pretrained_openvino_cpu` | Directory fot OpenVINO IR model. |

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
Argument `--complementary_align` allows to **add** new classes and images to existing alignment data,
`--complementary_train` uses existing embeddings for classifiers training, `--complementary` means `--complementary_align` and `--complementary_train` both.

So adding some new persons to classifiers from another input directory looks like this: 
```bash
python prepare.py --input_dir ./data/another_input_dir --complementary
```
Argument `--clarified` says that alignment input data is from servings images that was clarified, so this images will not detect for faces and uses as is and alignment will be complementary (key `--complementary_align` is not needed):
```bash
python prepare.py --input_dir ./data/clarify --clarified
```
Alignment can be executed by parts, use argument `--align_images_limit`:
```bash
python prepare.py --align_images_limit 100
```

It's available to skip alignment `--skip_alignment` or skip training `--skip_training`.

If recognizing process is running, this script will reload classifiers for running process (if `--skip_training` is not specified).

Classifiers can be uploaded to catalog's model right after successful training with name and version specified in arguments `--model-name` and `--model-version`.

### Download classifiers

Remote build classifiers can be downloaded with the following script:
```bash
python download_classifiers.py https://dev.kibernetika.io/api/v0.2/workspace/svod/mlmodel/svod-rcgn-3/versions/1.0.0/download/model-svod-rcgn-3-1.0.0.tar
```
Also you can specify classifiers dir (where to extract to) with argument `--classifiers_dir`

### Export classifiers to catalog model

Trained classifiers with attendant data can be exported to models catalog (if kdataset is installed or script is going to be run in kibernetika task:
```bash
python export_classifiers.py svod-faces 1.0.0
```
Classifiers dir (where to export from) can be set with argument `--classifiers_dir`

### Camera and recognizing

Open window with camera's video stream and draw labeled boxes for recognized faces: 

```bash
python video.py
```

Default video source is main web-camera, but it's can be redefined with argument `--video_source`.
Use argument `--video_async` for asynchronous video rendering (by default **each video's frame** waits for faces recognition, detection and add boxes and labels to this frame).

If any face has been recognized during 5 seconds (period can be changed with `--notify_face_detection_period`)
in more than 50% frames (default probability value 0.5 can be changed with `--notify_face_detection_prob`)
notification will be made.

Default notification makes to stdout.

Also notifications can be send to Slack channel if `--notify_slack_token` and `--notify_slack_channel` are filled.
Slack channel sets without leading `#`.


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
