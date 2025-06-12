YOLOv12
YOLOv12: Attention-Centric Real-Time Object Detectors
Yunjie Tian1, Qixiang Ye2, David Doermann1

1 University at Buffalo, SUNY, 2 University of Chinese Academy of Sciences.


Comparison with popular methods in terms of latency-accuracy (left) and FLOPs-accuracy (right) trade-offs

arXiv Hugging Face Demo Open In Colab Kaggle Notebook LightlyTrain Notebook deploy Openbayes

Updates
2025/06/04: YOLOv12's instance segmentation models are released, see code.

2025/04/15: Pretrain a YOLOv12 model with LightlyTrain, a novel framework that lets you pretrain any computer vision model on your unlabeled data, with YOLOv12 support. Here is also a Colab tutorial!

2025/03/18: Some guys are interested in the heatmap. See this issue.

2025/03/09: YOLOv12-turbo is released: a faster YOLOv12 version.

2025/02/24: Blogs: ultralytics, LearnOpenCV. Thanks to them!

2025/02/22: YOLOv12 TensorRT CPP Inference Repo + Google Colab Notebook.

2025/02/22: Android deploy / TensorRT-YOLO accelerates yolo12. Thanks to them!

2025/02/21: Try yolo12 for classification, oriented bounding boxes, pose estimation, and instance segmentation at ultralytics. Please pay attention to this issue. Thanks to them!

2025/02/20: Any computer or edge device? / ONNX CPP Version. Thanks to them!

2025/02/20: Train a yolov12 model on a custom dataset: Blog and Youtube. / Step-by-step instruction. Thanks to them!

2025/02/19: arXiv version is public. Demo is available (try Demo2 Demo3 if busy).

Abstract
Main Results
Turbo (default):

Model (det)	size
(pixels)	mAPval
50-95	Speed (ms)
T4 TensorRT10
params
(M)	FLOPs
(G)
YOLO12n	640	40.4	1.60	2.5	6.0
YOLO12s	640	47.6	2.42	9.1	19.4
YOLO12m	640	52.5	4.27	19.6	59.8
YOLO12l	640	53.8	5.83	26.5	82.4
YOLO12x	640	55.4	10.38	59.3	184.6
v1.0:

Model (det)	size
(pixels)	mAPval
50-95	Speed (ms)
T4 TensorRT10
params
(M)	FLOPs
(G)
YOLO12n	640	40.6	1.64	2.6	6.5
YOLO12s	640	48.0	2.61	9.3	21.4
YOLO12m	640	52.5	4.86	20.2	67.5
YOLO12l	640	53.7	6.77	26.4	88.9
YOLO12x	640	55.2	11.79	59.1	199.0
Instance segmentation:

Model (seg)	size
(pixels)	mAPbox
50-95	mAPmask
50-95	Speed (ms)
T4 TensorRT10
params
(M)	FLOPs
(B)
YOLOv12n-seg	640	39.9	32.8	1.84	2.8	9.9
YOLOv12s-seg	640	47.5	38.6	2.84	9.8	33.4
YOLOv12m-seg	640	52.4	42.3	6.27	21.9	115.1
YOLOv12l-seg	640	54.0	43.2	7.61	28.8	137.7
YOLOv12x-seg	640	55.2	44.2	15.43	64.5	308.7
Installation
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl
conda create -n yolov12 python=3.11
conda activate yolov12
pip install -r requirements.txt
pip install -e .
Validation
yolov12n yolov12s yolov12m yolov12l yolov12x

from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}.pt')
model.val(data='coco.yaml', save_json=True)
Training
from ultralytics import YOLO

model = YOLO('yolov12n.yaml')

# Train the model
results = model.train(
  data='coco.yaml',
  epochs=600, 
  batch=256, 
  imgsz=640,
  scale=0.5,  # S:0.9; M:0.9; L:0.9; X:0.9
  mosaic=1.0,
  mixup=0.0,  # S:0.05; M:0.15; L:0.15; X:0.2
  copy_paste=0.1,  # S:0.15; M:0.4; L:0.5; X:0.6
  device="0,1,2,3",
)

# Evaluate model performance on the validation set
metrics = model.val()

# Perform object detection on an image
results = model("path/to/image.jpg")
results[0].show()
Prediction
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}.pt')
model.predict()
Export
from ultralytics import YOLO

model = YOLO('yolov12{n/s/m/l/x}.pt')
model.export(format="engine", half=True)  # or format="onnx"
Demo
python app.py
# Please visit http://127.0.0.1:7860
Acknowledgement
The code is based on ultralytics. Thanks for their excellent work!

Citation
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
