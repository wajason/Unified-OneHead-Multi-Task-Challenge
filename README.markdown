# Unified-OneHead Multi-Task Challenge

## Introduction
This project is part of the **Unified-OneHead Multi-Task Challenge**, which aims to develop a single head (2-3 layers) neural network to simultaneously perform **Object Detection**, **Semantic Segmentation**, and **Image Classification**. The goal is to mitigate catastrophic forgetting, ensuring each task's performance drops by no more than 5% compared to its single-task baseline.

### Key Requirements:
- **Tasks**: Single head outputs for Mini-COCO-Det (detection), Mini-VOC-Seg (segmentation), and Imagenette-160 (classification).
- **Datasets**: Three mini-datasets, each with 300 images (240 train / 60 val), total size < 120MB:
  - Mini-COCO-Det: 45MB, COCO 2017 (10 classes), COCO JSON annotations.
  - Mini-VOC-Seg: 30MB, PASCAL VOC 2012, PNG masks.
  - Imagenette-160: 25MB, Imagenette v2, folder/label structure.
- **Hardware**: Free Google Colab GPU (T4 or V100), training time ≤ 2 hours.
- **Deliverables**: `colab.ipynb`, `report.md`, `llm_dialogs.zip`, submitted via GitHub Pull Request.

This README tracks the project progress, starting with dataset preparation. It will be updated as the project advances.

## Dataset Preparation

### Overall Structure
The datasets are organized as follows:
```
data/
  mini_coco_det/{train,val}
  mini_voc_seg/{train,val}
  imagenette_160/{train,val}
```

### Imagenette-160
- **Source**: Downloaded from [fastai/imagenette](https://github.com/fastai/imagenette) (`imagenette2-160.tgz`).
- **Details**: Contains 300 images (240 train / 60 val) across 10 classes (`tench`, `English springer`, `cassette player`, `chain saw`, `church`, `French horn`, `garbage truck`, `gas pump`, `golf ball`, `parachute`).
- **Structure**:
  ```
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
  ```
  - `train/` has 240 images (24 per class).
  - `val/` has 60 images (6 per class).
- **Size**: Approximately 2.79MB (uncompressed). Note: The PDF specifies 25MB, likely a typo; expected size based on ~7KB per image is ~2.1MB.
- **Annotation**: Folder/label structure, with class names as labels.

### Next Steps
- **Mini-COCO-Det**: Prepare 300 images (10 classes, 45MB) from COCO 2017 with COCO JSON annotations.
- **Mini-VOC-Seg**: Prepare 300 images (30MB) from PASCAL VOC 2012 with PNG masks.
- Updates will be added as datasets are completed.

## Future Updates
This README will be updated with:
- Model architecture and training details.
- Forgetting mitigation strategies.
- Performance evaluation results.
- Instructions for running the code and reproducing results.