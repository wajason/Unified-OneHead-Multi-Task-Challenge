# Unified-OneHead Multi-Task Challenge

## Introduction
This project is part of the **Unified-OneHead Multi-Task Challenge**, which aims to develop a single head (2-3 layers) neural network to simultaneously perform **Object Detection**, **Semantic Segmentation**, and **Image Classification**. The goal is to mitigate catastrophic forgetting, ensuring each task's performance drops by no more than 5% compared to its single-task baseline.

### Key Requirements:
- **Tasks**: Single head outputs for Mini-COCO-Det (detection), Mini-VOC-Seg (segmentation), and Imagenette-160 (classification).
- **Datasets**: Three mini-datasets, each with 300 images (240 train / 60 val), total size < 120MB:
  - Mini-COCO-Det: 45MB, COCO 2017 (10 classes), COCO JSON annotations.
  - Mini-VOC-Seg: 33MB, PASCAL VOC 2012, PNG masks.
  - Imagenette-160: 2.79MB, Imagenette v2, folder/label structure.
- **Hardware**: Free Google Colab GPU (T4 or V100), training time ≤ 2 hours.
- **Deliverables**: `colab.ipynb`, `report.md`, `llm_dialogs.zip`, submitted via GitHub Pull Request.

This README tracks the project progress, starting with dataset preparation. It will be updated as the project advances.

## Dataset Preparation

### Overall Structure
The datasets are organized as follows:
data/
mini_coco_det/
train/
data/  # 240 images
labels.json  # COCO JSON annotations
val/
data/  # 60 images
labels.json  # COCO JSON annotations
mini_voc_seg/
train/  # 240 images with PNG masks
val/    # 60 images with PNG masks
imagenette_160/
train/  # 240 images across 10 classes
val/    # 60 images across 10 classes


### Mini-COCO-Det
- **Source**: Derived from COCO 2017 dataset (`train2017` and `val2017` splits).
- **Details**: Contains 300 images (240 train / 60 val) with 10 classes (`person`, `car`, `dog`, `cat`, `chair`, `table`, `book`, `bottle`, `cup`, `bird`).
- **Structure**:
data/
mini_coco_det/
train/
data/
000000000025.jpg
000000000030.jpg
...
labels.json
val/
data/
000000000049.jpg
000000000064.jpg
...
labels.json


- `train/` has 240 images with corresponding COCO JSON annotations in `labels.json`.
- `val/` has 60 images with corresponding COCO JSON annotations in `labels.json`.
- **Size**: Approximately 47.8MB (uncompressed).
- **Annotation**: COCO JSON format, containing bounding box annotations for the 10 specified classes.

### Mini-VOC-Seg
- **Source**: Derived from PASCAL VOC 2012 dataset.
- **Details**: Contains 300 images (240 train / 60 val) with semantic segmentation masks for various classes (e.g., `person`, `car`, `dog`, etc.).
- **Structure**:
data/
mini_voc_seg/
train/
JPEGImages/
2007_000032.jpg
2007_000039.jpg
...
SegmentationClass/
2007_000032.png
2007_000039.png
...
val/
JPEGImages/
2007_000042.jpg
2007_000061.jpg
...
SegmentationClass/
2007_000042.png
2007_000061.png
...


- `train/` has 240 images with corresponding PNG masks.
- `val/` has 60 images with corresponding PNG masks.
- **Size**: Approximately 33MB (uncompressed).
- **Annotation**: PNG masks, where each pixel value corresponds to a class label.

### Imagenette-160
- **Source**: Downloaded from [fastai/imagenette](https://github.com/fastai/imagenette) (`imagenette2-160.tgz`).
- **Details**: Contains 300 images (240 train / 60 val) across 10 classes (`tench`, `English springer`, `cassette player`, `chain saw`, `church`, `French horn`, `garbage truck`, `gas pump`, `golf ball`, `parachute`).
- **Structure**:
data/
imagenette_160/
train/
tench/
English_springer/
cassette_player/
chain_saw/
church/
French_horn/
garbage_truck/
gas_pump/
golf_ball/
parachute/
val/
tench/
English_springer/
cassette_player/
chain_saw/
church/
French_horn/
garbage_truck/
gas_pump/
golf_ball/
parachute/


- `train/` has 240 images (24 per class).
- `val/` has 60 images (6 per class).
- **Size**: Approximately 2.79MB (uncompressed). Note: The PDF specifies 25MB, likely a typo; expected size based on ~7KB per image is ~2.1MB.
- **Annotation**: Folder/label structure, with class names as labels.

### Summary of Datasets
| Dataset        | Task                  | Size    | Images (Train/Val) | Classes | Annotation Format      |
|----------------|-----------------------|---------|--------------------|---------|------------------------|
| Mini-COCO-Det  | Object Detection      | 47.8MB  | 300 (240/60)       | 10      | COCO JSON              |
| Mini-VOC-Seg   | Semantic Segmentation | 33MB    | 300 (240/60)       | Varies  | PNG masks              |
| Imagenette-160 | Image Classification  | 2.79MB  | 300 (240/60)       | 10      | Folder/label structure |

Total size: 47.8MB + 33MB + 2.79MB = 83.59MB, which is under the 120MB limit.

## Next Steps
- **Model Implementation**: Start coding in `colab.ipynb` to design a single-head neural network for the three tasks.
- **Training**: Train the model on Google Colab GPU (T4 or V100) within 2 hours.
- **Evaluation**: Measure performance and mitigate catastrophic forgetting.
- **Documentation**: Update README with model details, training process, and results.

## Future Updates
This README will be updated with:
- Model architecture and training details.
- Forgetting mitigation strategies.
- Performance evaluation results.
- Instructions for running the code and reproducing results.
