# Unified Single-Head Multi-Task Learning Framework with Catastrophic Forgetting Mitigation

## Overview
This repository presents a unified single-head multi-task learning framework designed to concurrently address Semantic Segmentation (seg), Object Detection (det), and Image Classification (cls) tasks within a shared backbone architecture. The framework is engineered to optimize computational efficiency and mitigate catastrophic forgetting, a critical challenge in sequential multi-task learning scenarios.

Our model leverages EfficientNet-B0 as the backbone feature extractor, pre-trained on the ImageNet dataset, augmented with a Feature Pyramid Network (FPN), and employs Knowledge Distillation (KD) to mitigate catastrophic forgetting across sequential task training, and our model framework is trained and validated on the following datasets. 

The framework is optimized for parameter efficiency (Total parameters: 4,175,137 < 8M parameters) and evaluated for performance stability within a 5% drop from baseline metrics, we can know that on Training Schedule.

## Technical Implementation
### 1. Architecture Design
The model architecture is structured as follows:
- **Backbone**: EfficientNet-B0, pre-trained on ImageNet, extracts multi-scale features at strides 8, 16, and 32.
- **Neck**: FPN processes backbone outputs into a 128-channel feature map, with an additional max-pooled layer for enhanced resolution.
- **Shared Layer**: A 3x3 convolution reduces channels to 64, serving as the common feature space.
- **Task-Specific Heads**:
  - **Segmentation**: 1x1 convolution followed by bilinear upsampling to 512x512, outputting 21 classes.
  - **Detection**: 1x1 convolution generating 6 channels (cx, cy, w, h, conf, class_id) on a 16x16 grid.
  - **Classification**: Global average pooling and a linear layer for 10 classes.

Total parameter count is approximately 4.9M, adhering to the < 8M constraint.

### 2. Training Protocol
- **Schedule**: Sequential training across three tasks (seg, det, cls), with 5 epochs per task (total 15 epochs).
- **Optimizer**: AdamW with initial learning rate 0.0008, weight decay 1e-4.
- **Scheduler**: CosineAnnealingLR over the total epochs.
- **Batch Size**: 4, with input resolution 512x512.
- **Dataset**: Mini-VOC (seg), Mini-COCO (det), Imagenette-160 (cls).

### 3. Catastrophic Forgetting Mitigation
Catastrophic forgetting is addressed using Knowledge Distillation (KD), applied post-first stage. The KD loss is defined as:
\[ L_{KD} = \lambda \cdot T^2 \cdot KL(\text{soft}(\mathbf{z}_s / T) || \text{soft}(\mathbf{z}_t / T)) \]
where \(\mathbf{z}_s\) and \(\mathbf{z}_t\) are student and teacher logits, \(T = 2.0\) is the temperature, and \(\lambda = 1.0\) is the weighting factor. The teacher model, initialized from the previous stage, provides soft targets to stabilize prior task performance.

### 4. Evaluation Metrics
- **Segmentation**: Mean Intersection over Union (mIoU) on 21 classes.
- **Detection**: Mean Average Precision (mAP) with IoU threshold 0.5.
- **Classification**: Top-1 and Top-5 accuracy on 10 classes.
- **Objective**: Performance drop < 5% from baseline, with bonus points for metrics exceeding baseline.

## Performance Results
Preliminary results (subject to final execution):
- **Seg mIoU**: [TBD, e.g., 0.45] (Baseline: [TBD, e.g., 0.47])
- **Det mAP**: [TBD, e.g., 0.20] (Baseline: [TBD, e.g., 0.22])
- **Cls Top-1**: [TBD, e.g., 0.52] (Baseline: [TBD, e.g., 0.51])
- **Training Time**: Projected ≤ 2 hours on GPU.
- **Inference Time**: Targeted < 150 ms per image.

## Contributing
Contributions to enhance the framework are welcome. To contribute:
1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make changes and commit them (`git commit -m "Add feature"`).
4. Push to the branch (`git push origin feature-branch`).
5. Open a Pull Request with a clear description of changes.

Please ensure code adheres to PEP 8 guidelines and includes tests or documentation updates.

## License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments
- **mlxtend Community**: For providing robust association rule mining tools.
- **Dataset Providers**: For enabling this analysis with high-quality datasets.
- **Inspiration**: Drawn from real-world applications in multi-task learning and e-commerce recommendation systems.

## Future Work
- Refine the detection loss function to improve mAP.
- Validate inference latency on diverse hardware.
- Extend KD to incorporate replay buffers for enhanced stability.

## Contact
For questions or collaboration, please contact [Your Email].

---