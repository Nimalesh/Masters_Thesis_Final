# Explainable Attention-Guided Multi-Task Learning for Breast Ultrasound

[![Python 3.9](https://img.shields.io/badge/python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch 1.13.0](https://img.shields.io/badge/PyTorch-1.13.0-EE4C2C.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official PyTorch implementation for the thesis:

**"Attention-Guided Multi-Task Learning for Explainable Breast Ultrasound Segmentation and Classification"**

---

## 📌 Overview

Breast Ultrasound (BUS) imaging plays a crucial role in early breast cancer diagnosis. However, challenges such as class imbalance, noisy ultrasound artifacts, and limited explainability often reduce the effectiveness of conventional deep learning approaches.

This repository presents an **Explainable Attention-Guided Multi-Task Learning Framework** that simultaneously performs:

- **Breast lesion segmentation**
- **Breast lesion classification**
- **Attention-based explainability**

The proposed architecture is designed to improve feature discrimination, reduce task interference, address class imbalance, and provide clinically interpretable predictions.

---

## 🏗️ Architecture Overview

The framework consists of four major components:

### 1. EfficientNet-B6 Encoder with CBAM Attention

- EfficientNet-B6 serves as the backbone feature extractor.
- Convolutional Block Attention Module (CBAM) enhances:
  - Channel-wise feature importance
  - Spatial feature localization
- Improves robustness against ultrasound noise and imaging artifacts.

### 2. Latent-Space Augmentation

To mitigate severe class imbalance, especially for malignant lesions:

- Augmentation is performed directly within the latent representation.
- Gaussian perturbation is applied at the bottleneck layer.
- Preserves segmentation mask integrity while increasing minority-class diversity.
- Prevents overfitting to limited malignant samples.

### 3. Nested UNet++ Decoder

The decoder utilizes:

- Dense skip connections
- Multi-scale feature aggregation
- Progressive feature refinement

Benefits:

- Improved boundary delineation
- Better lesion localization
- Enhanced segmentation accuracy

### 4. Multi-Scale Deep Supervision

Intermediate decoder outputs are supervised using resized ground-truth masks.

Advantages include:

- Stable gradient propagation
- Reduced semantic collapse
- Improved convergence
- Better interaction between segmentation and classification tasks

---

## ✨ Key Features

- Multi-task learning for segmentation and classification
- CBAM attention-guided feature extraction
- Latent-space augmentation for class imbalance handling
- UNet++ dense skip architecture
- Deep supervision at multiple decoder scales
- Explainable feature learning
- End-to-end trainable framework
- Medical imaging focused design

---

## ⚙️ Requirements

Experiments were conducted using:

| Component | Version |
|------------|------------|
| Python | 3.9 |
| PyTorch | 1.13.0 |
| CUDA | 11.x |
| GPU | NVIDIA T4 |

---

## 📥 Installation

Clone the repository:

```bash
git clone https://github.com/Nimalesh/Thesis_Phase1.git
cd Thesis_Phase1
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 📂 Dataset Preparation

Due to medical data privacy restrictions, the original BUS dataset cannot be publicly released.

Users should implement their own PyTorch Dataset and DataLoader.

### Input Images

- RGB ultrasound images
- Resize to:

```text
256 × 256
```

- Normalize using ImageNet statistics:

```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

### Segmentation Masks

- Binary masks
- Values:

```text
0 = Background
1 = Lesion
```

- Size:

```text
256 × 256
```

### Classification Labels

Example:

| Label | Class |
|---------|---------|
| 0 | Normal |
| 1 | Benign |
| 2 | Malignant |

---

## 🚀 Training

The training pipeline employs:

- AdamW optimizer
- Cosine Annealing Learning Rate Scheduler
- Mixed-task optimization
- Gradient Accumulation (2 steps)

This simulates an effective batch size of 16 while reducing GPU memory consumption.

### Start Training

```bash
python train.py
```

---

## 🔍 Inference

During evaluation:

- Deep supervision heads are automatically disabled.
- Latent augmentation is skipped.
- Only final classification logits and segmentation masks are produced.

### Example

```python
import torch
from models.network import ExplainableMultiTaskNet

# Load model
model = ExplainableMultiTaskNet(num_classes=3)

model.load_state_dict(
    torch.load("explainable_multitask_bus.pth")
)

model.eval()

# Example input
x = torch.randn(1, 3, 256, 256)

with torch.no_grad():
    cls_logits, seg_mask = model(x)

print("Classification Logits:", cls_logits.shape)
print("Predicted Segmentation Mask:", seg_mask.shape)
```

Expected outputs:

```text
Classification Logits: [1, 3]
Predicted Segmentation Mask: [1, 1, 256, 256]
```

---

## 📊 Loss Functions

### Segmentation Branch

Combination of:

- Binary Cross-Entropy Loss (BCE)
- Dice Loss
- Deep Supervision Loss

### Classification Branch

- Cross-Entropy Loss

### Total Objective

```text
L_total =
λ1 L_segmentation
+
λ2 L_classification
+
λ3 L_deep_supervision
```

where:

- λ₁ = segmentation weight
- λ₂ = classification weight
- λ₃ = deep supervision weight

---

## 📝 Implementation Notes

### Latent Gaussian Perturbation

The public implementation includes:

```text
Latent Gaussian Perturbation
```

as described in Thesis Equation (3.9).

This strategy:

- Generates synthetic latent representations
- Improves minority-class generalization
- Preserves spatial correspondence for segmentation

### Manifold MixUp

The original thesis also investigated:

```text
Manifold MixUp
```

(Thesis Equation 3.6)

However, it is intentionally omitted from this repository to maintain:

- Simplicity
- Readability
- Reproducibility

The included latent perturbation mechanism provides comparable anti-imbalance benefits.

---

## 📈 Applications

Potential clinical applications include:

- Breast cancer screening
- Computer-aided diagnosis (CAD)
- Lesion localization
- Explainable AI in radiology
- Clinical decision support systems

---

## 📚 Acknowledgements

This implementation was inspired by several open-source projects and research works:

- UNet++
- EfficientNet
- CBAM
- Tsong-UNetPP
- SDL-BR-Net
- ASFF
- ETF-YOLO11

The authors gratefully acknowledge the contributions of the medical imaging and deep learning research communities.

---

## 🎓 Citation

If you use this work in your research, please cite:

```bibtex
@mastersthesis{Elangovan2026,
  author       = {Nimalesh Elangovan},
  title        = {Attention-Guided Multi-Task Learning for Explainable Breast Ultrasound Segmentation and Classification},
  school       = {Indian Institute of Technology Madras, Zanzibar Campus},
  year         = {2026}
}
```

---

## 📜 License

This project is released under the MIT License.

See the LICENSE file for details.

---

## 👨‍💻 Author

**Nimalesh Elangovan**

M.Tech in Data Science and Artificial Intelligence  
Indian Institute of Technology Madras, Zanzibar Campus

For academic collaborations, discussions, or questions regarding the implementation, feel free to open an issue in this repository.

---