YOLOv12: Attention-Centric Real-Time Object Detection and Segmentation
Authors: Yunjie Tian (University at Buffalo, SUNY), Qixiang Ye (UCAS), David Doermann (University at Buffalo, SUNY)
Paper: arXiv:2502.12524
Demo: Try it on Hugging Face | Google Colab | Kaggle Notebook

🚀 Introduction
YOLOv12 introduces an attention-based architecture for real-time object detection and segmentation, pushing the boundaries of both accuracy and efficiency across a range of deployment scenarios—from edge devices to data centers. Building upon prior YOLO variants, YOLOv12 integrates advanced attention mechanisms and design principles optimized for latency-accuracy and FLOPs-accuracy trade-offs.

📊 Performance Overview
📍 Detection (Turbo Release)
Model	Input Size	mAP@50–95	Inference Time (ms)	Params (M)	FLOPs (G)
YOLOv12n	640	40.4	1.60	2.5	6.0
YOLOv12s	640	47.6	2.42	9.1	19.4
YOLOv12m	640	52.5	4.27	19.6	59.8
YOLOv12l	640	53.8	5.83	26.5	82.4
YOLOv12x	640	55.4	10.38	59.3	184.6

🧩 Instance Segmentation
Model	Input Size	mAP (Box)	mAP (Mask)	Time (ms)	Params (M)	FLOPs (B)
YOLOv12n-seg	640	39.9	32.8	1.84	2.8	9.9
YOLOv12s-seg	640	47.5	38.6	2.84	9.8	33.4
YOLOv12m-seg	640	52.4	42.3	6.27	21.9	115.1
YOLOv12l-seg	640	54.0	43.2	7.61	28.8	137.7
YOLOv12x-seg	640	55.2	44.2	15.43	64.5	308.7

📦 Installation
bash
Copy
Edit
# Download FlashAttention for performance boosts
wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu11torch2.2cxx11abiFALSE-cp311-cp311-linux_x86_64.whl

# Set up virtual environment
conda create -n yolov12 python=3.11
conda activate yolov12

# Install dependencies
pip install -r requirements.txt
pip install -e .
🧪 Validation
python
Copy
Edit
from ultralytics import YOLO

# Load and validate the model
model = YOLO("yolov12s.pt")
model.val(data="coco.yaml", save_json=True)
🏋️ Training
python
Copy
Edit
from ultralytics import YOLO

# Initialize model config
model = YOLO("yolov12n.yaml")

# Train with custom hyperparameters
results = model.train(
    data="coco.yaml",
    epochs=600,
    batch=256,
    imgsz=640,
    scale=0.5,
    mosaic=1.0,
    mixup=0.0,
    copy_paste=0.1,
    device="0,1,2,3",
)
🔍 Inference
python
Copy
Edit
from ultralytics import YOLO

model = YOLO("yolov12m.pt")
results = model("path/to/image.jpg")
results[0].show()
📤 Export Models
python
Copy
Edit
from ultralytics import YOLO

# Export to ONNX or TensorRT Engine
model = YOLO("yolov12x.pt")
model.export(format="engine", half=True)
🖥️ Web Demo
To launch a local demo:

bash
Copy
Edit
python app.py
# Visit http://127.0.0.1:7860 in your browser
🔧 Use Cases
Object Detection (COCO, custom datasets)

Instance Segmentation

Oriented Bounding Boxes

Pose Estimation

Image Classification

📚 Citation
If you use YOLOv12 in your research or application, please cite the following:

bibtex
Copy
Edit
@article{tian2025yolov12,
  title={YOLOv12: Attention-Centric Real-Time Object Detectors},
  author={Tian, Yunjie and Ye, Qixiang and Doermann, David},
  journal={arXiv preprint arXiv:2502.12524},
  year={2025}
}
🙏 Acknowledgments
YOLOv12 builds upon the excellent foundation laid by Ultralytics. We also thank the contributors from the open-source community who supported testing and deployment across multiple platforms (e.g., TensorRT, Android, OpenBayes).
