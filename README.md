# 🛣️ Road Damage Detection, Classification & Severity Estimation

<p align="center">

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?style=for-the-badge\&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?style=for-the-badge\&logo=pytorch)
![YOLOv8](https://img.shields.io/badge/YOLOv8n-Custom%20Architecture-00FFFF?style=for-the-badge)
![CUDA](https://img.shields.io/badge/CUDA-GPU%20Training-76B900?style=for-the-badge\&logo=nvidia)
![Kaggle](https://img.shields.io/badge/Training-Kaggle-20BEFF?style=for-the-badge\&logo=kaggle)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-5C3EE8?style=for-the-badge\&logo=opencv)
![RDD2022](https://img.shields.io/badge/Dataset-RDD2022-orange?style=for-the-badge)

</p>

<p align="center">
<b>🚧 Intelligent Road Damage Detection with SCConv + EMA + P2 Multi-Scale Detection</b>
</p>

<p align="center">
Detect road damage • Classify damage type • Estimate severity • Calculate Road Health Index
</p>

---

# 🌟 Project Overview

Road infrastructure deteriorates because of cracks, potholes, surface failures, and other forms of pavement damage. Manual road inspection is time-consuming, expensive, and difficult to scale.

This project develops a deep-learning-based road damage analysis system using a customized **YOLOv8n** architecture.

The model architecture is:

> **YOLOv8n + SCConv + EMA + P2**

The model is designed to improve the detection of road-damage objects at different scales, with particular attention to small and fine-grained damage such as thin cracks.

The architecture introduces three major modifications:

* 🧩 **SCConv — Spatial and Channel Reconstruction Convolution**
* 🎯 **EMA — Efficient Multi-scale Attention**
* 🔍 **P2 Detection Head — High-resolution small-object detection**

The final detector uses four feature scales:

```text
P2 → 160 × 160
P3 → 80 × 80
P4 → 40 × 40
P5 → 20 × 20
```

For a `640 × 640` input image, P2 operates at **stride 4**, allowing the network to preserve considerably more spatial information than coarse detection layers.

---

# 🚀 Key Features

| Feature                      | Description                                       |
| ---------------------------- | ------------------------------------------------- |
| 🧠 **YOLOv8n**               | Lightweight one-stage object detector             |
| 🧩 **SCConv**                | Spatial and channel feature reconstruction        |
| 🎯 **EMA**                   | Efficient attention for important feature regions |
| 🔍 **P2 Head**               | High-resolution detection for small road damage   |
| 🔄 **Bidirectional Fusion**  | Top-down and bottom-up feature propagation        |
| 🛣️ **RDD2022**              | Road damage detection dataset                     |
| 🏷️ **5 Classes**            | Crack, pothole, and other road-damage categories  |
| 📊 **Multi-scale Detection** | P2/P3/P4/P5 prediction                            |
| 📈 **Severity Estimation**   | Detection-based derived severity score            |
| 🏥 **RHI**                   | Road Health Index derived from severity           |
| ⚡ **GPU Training**           | CUDA-enabled PyTorch training                     |
| 💾 **YOLO `.pt`**            | Standard PyTorch checkpoint output                |

---

# 🧱 System Architecture

The complete system consists of four conceptual stages:

```text
                📷 ROAD IMAGE
                     │
                     ▼
        ┌───────────────────────────┐
        │     YOLOv8n Backbone      │
        │                           │
        │ Conv → C2f → Conv → C2f  │
        │      ↓                    │
        │    SCConv                 │
        │      ↓                    │
        │ Conv → C2f → SCConv       │
        │      ↓                    │
        │ SPPF → P5                 │
        └─────────────┬─────────────┘
                      │
                      ▼
             🔄 FEATURE NECK
                      │
          ┌───────────┴───────────┐
          │                       │
     Top-Down Path          Bottom-Up Path
          │                       │
          ▼                       ▼
       P5 → P4                P2 → P3
       P4 → P3                P3 → P4
       P3 → P2                P4 → P5
          │                       │
          └───────────┬───────────┘
                      │
                 🎯 EMA Attention
                      │
                      ▼
             P2 / P3 / P4 / P5
                      │
                      ▼
             🧠 DETECTION HEAD
                      │
                      ▼
          ┌─────────────────────┐
          │ Damage Class        │
          │ Confidence          │
          │ Bounding Box        │
          └──────────┬──────────┘
                     │
                     ▼
             📊 SEVERITY ENGINE
                     │
                     ▼
              🏥 ROAD HEALTH
                 INDEX (RHI)
```

---

# 🧠 1. YOLOv8n Backbone

The base network is a lightweight **YOLOv8n-style backbone**.

The input image is resized to:

$$
640 \times 640 \times 3
$$

The backbone progressively reduces spatial resolution while increasing the number of feature channels.

## Backbone Progression

| Stage |  Resolution | Channels | Purpose                                 |
| ----- | ----------: | -------: | --------------------------------------- |
| Input | `640 × 640` |        3 | RGB image                               |
| P1    | `320 × 320` |       16 | Initial feature extraction              |
| P2    | `160 × 160` |       32 | High-resolution features                |
| P3    |   `80 × 80` |       64 | Small/medium damage features            |
| P4    |   `40 × 40` |      128 | Medium-scale semantic features          |
| P5    |   `20 × 20` |      256 | High-level semantic/contextual features |

The architecture uses **C2f blocks** for efficient feature extraction and gradient propagation.

---

# 🧩 2. SCConv Modification

## What is SCConv?

SCConv is used in this project to improve feature representation through:

* Spatial reconstruction
* Channel reconstruction

In the implemented architecture, SCConv is inserted at the **P3 and P4 backbone stages**.

```text
Feature Map
     │
     ▼
  3×3 Conv
     │
     ▼
 BatchNorm
     │
     ▼
   SiLU
     │
     ▼
    SRU
     │
     ▼
    CRU
     │
     ▼
Enhanced Feature Map
```

---

## 🔹 Spatial Reconstruction Unit — SRU

The SRU first applies Group Normalization.

A feature importance value is calculated from the normalized feature map:

$$
W = \sigma\left(\operatorname{Mean}_{H,W}(X_{\text{norm}})\right)
$$

where:

* $X_{\text{norm}}$ = normalized feature map
* $\sigma$ = sigmoid function
* $W$ = feature importance

A threshold of:

$$
T = 0.5
$$

is used in the implementation to generate a gate.

The feature map is divided conceptually into:

```text
          Feature Map
              │
       ┌──────┴──────┐
       ▼             ▼
 High-information  Low-information
    features          features
       │                 │
       │             α × feature
       └───────┬─────────┘
               ▼
       Reconstructed map
```

The implemented reconstruction is:

$$
X_{\text{SRU}} = X_{\text{high}} + \alpha X_{\text{low}}
$$

with:

$$
\alpha = 0.5
$$

for the SCConv configuration used in the model.

### 🎯 Why SRU is useful

Road images contain many irrelevant regions:

* Road texture
* Shadows
* Lane markings
* Vehicles
* Vegetation
* Background objects

The SRU attempts to separate stronger and weaker feature responses so that useful road-damage information is preserved.

---

# 🔹 Channel Reconstruction Unit — CRU

After SRU, the feature representation is passed through the CRU.

The implementation performs:

```text
Input
  │
  ▼
1×1 Channel Reduction
  │
  ▼
BatchNorm
  │
  ▼
SiLU
  │
  ▼
1×1 Channel Expansion
  │
  ▼
Residual Addition
```

The CRU first reduces the number of channels:

$$
C' = \max(8, C/2)
$$

Then the channels are expanded back to the original number.

A residual connection is used:

$$
Y = F(X) + X
$$

This helps retain the original information while reconstructing the channel representation.

---

# 🧩 Why SCConv is Used

SCConv is positioned at:

```text
P3 → SCConv

P4 → SCConv
```

This provides feature reconstruction at intermediate spatial scales.

The intended benefit is:

> Better feature representation with reduced redundancy, especially for complex road-damage patterns.

---

# 🎯 3. EMA Attention Modification

## What is EMA?

EMA stands for **Efficient Multi-scale Attention**.

EMA is inserted in the **neck**, after feature fusion.

The project uses EMA at the following channel sizes:

```text
EMA(128)
EMA(64)
EMA(32)
EMA(64)
EMA(128)
EMA(256)
```

Therefore, EMA participates in both the top-down and bottom-up feature pathways.

---

# 🔬 EMA Working Principle

For an input:

$$
X \in \mathbb{R}^{B \times C \times H \times W}
$$

the channels are divided into groups.

The implementation uses a grouping factor of up to:

$$
G = 8
$$

where the actual number of groups is adjusted so that the channel count is divisible by the number of groups.

The grouped representation is:

$$
X_g \in
\mathbb{R}^{(B \times G) \times (C/G) \times H \times W}
$$

---

## 🔹 Horizontal Pooling

The model performs adaptive pooling along the width:

$$
X_h = \operatorname{Pool}_W(X_g)
$$

This retains height information while compressing the width.

This allows the attention mechanism to capture directional contextual information.

---

## 🔹 Vertical Pooling

The model also performs pooling along the height:

$$
X_w = \operatorname{Pool}_H(X_g)
$$

The two representations provide complementary directional information.

---

## 🔹 Attention Map

The implementation combines the pooled representations:

$$
A =
\sigma
\left(
\operatorname{Conv}_{1\times1}(X_h) + X_w
\right)
$$

where:

* $A$ = attention map
* $\operatorname{Conv}_{1\times1}$ = 1×1 convolution
* $\sigma$ = sigmoid function

---

## 🔹 Local Feature Extraction

A 3×3 convolution extracts local features:

$$
L = \operatorname{Conv}_{3\times3}(X_g)
$$

The attention map then weights these local features:

$$
Y = L \odot A
$$

where $\odot$ denotes element-wise multiplication.

Group normalization is applied and the result is reshaped back to the original channel structure.

Finally, a residual connection is used:

$$
Y_{\text{EMA}} = GN(L \odot A) + X
$$

---

# 🛣️ Why EMA is Useful for Road Damage

Road damage is often directional and irregular.

Examples:

```text
Longitudinal crack

────────────────────


Transverse crack

       │
       │
       │


Alligator crack

 /\__/\/\_
/_\_/\_/\_
```

The directional pooling operations in EMA provide contextual information along horizontal and vertical dimensions.

This can help the network emphasize important regions associated with cracks and other road damage.

---

# 🔍 4. P2 Detection Head

One of the most important modifications is the introduction of a **P2 detection scale**.

Standard three-scale detection generally uses:

```text
P3
P4
P5
```

This architecture uses:

```text
P2
P3
P4
P5
```

For a `640 × 640` input:

| Scale | Feature Map | Stride | Primary Role         |
| ----- | ----------: | -----: | -------------------- |
| P2    | `160 × 160` |      4 | Small/fine damage    |
| P3    |   `80 × 80` |      8 | Small/medium damage  |
| P4    |   `40 × 40` |     16 | Medium damage        |
| P5    |   `20 × 20` |     32 | Large damage/context |

---

## 🔎 Why P2 Matters

A thin crack may occupy only a small number of pixels.

If the image is repeatedly downsampled:

```text
640 × 640
    ↓
320 × 320
    ↓
160 × 160
    ↓
80 × 80
    ↓
40 × 40
    ↓
20 × 20
```

fine structures can become difficult to represent.

P2 preserves a:

$$
160 \times 160
$$

feature map.

Therefore, the model can make predictions using substantially higher-resolution spatial information.

This is particularly relevant to:

* Thin longitudinal cracks
* Thin transverse cracks
* Small potholes
* Small isolated damage
* Fragmented road defects

---

# 🔄 5. Bidirectional Feature Fusion

The neck of the model is not simply a conventional top-down pathway.

It contains:

## Top-Down Pathway

```text
P5
 ↓ Upsample
P4
 ↓ Upsample
P3
 ↓ Upsample
P2
```

## Bottom-Up Pathway

```text
P2
 ↓ Downsample
P3
 ↓ Downsample
P4
 ↓ Downsample
P5
```

This creates bidirectional information propagation.

---

# ⬇️ Top-Down Pathway

High-level semantic information from P5 is progressively transferred to higher-resolution feature maps.

```text
P5
 │
 ├── Upsample
 ▼
P4 + P5
 │
 ├── C2f
 ├── EMA
 ▼
Enhanced P4
 │
 ├── Upsample
 ▼
P3 + P4
 │
 ├── C2f
 ├── EMA
 ▼
Enhanced P3
 │
 ├── Upsample
 ▼
P2 + P3
 │
 ├── C2f
 ├── EMA
 ▼
Enhanced P2
```

This allows high-level semantic information to assist small-object detection.

---

# ⬆️ Bottom-Up Pathway

The enhanced P2 information is then propagated upward.

```text
Enhanced P2
     │
     ▼
  Downsample
     │
     ▼
P3 Fusion
     │
   C2f + EMA
     │
     ▼
  Downsample
     │
     ▼
P4 Fusion
     │
   C2f + EMA
     │
     ▼
  Downsample
     │
     ▼
P5 Fusion
     │
   C2f + EMA
```

This is important because the fine-grained information extracted at P2 does not remain isolated.

It can influence P3, P4, and P5 representations.

---

# 🎯 6. Final Detection Head

The final detector receives:

$$
[P2, P3, P4, P5]
$$

and predicts:

* Bounding boxes
* Class probabilities
* Confidence scores

The project uses **5 damage classes**.

---

# 🏷️ Damage Classes

| Class ID | Class              |
| -------: | ------------------ |
|        0 | Longitudinal crack |
|        1 | Transverse crack   |
|        2 | Alligator crack    |
|        3 | Other corruption   |
|        4 | Pothole            |

The model therefore performs multi-class object detection rather than simple binary road/no-road classification.

---

# 📊 Detection Output

For each detected object, the model provides:

```text
Class
Confidence
Bounding Box
```

Example:

```text
Damage: Pothole

Confidence: 0.91

Bounding Box:
x1 = 180
y1 = 220
x2 = 390
y2 = 410
```

These outputs are subsequently used by the project-defined severity calculation.

---

# 🏥 7. Severity Estimation

## Important Distinction

The severity value in this project is a **derived engineering/application metric** calculated from the YOLO detections.

It is **not a severity value directly predicted by the YOLOv8 detection head**.

The detector provides:

```text
Damage class
     +
Confidence
     +
Bounding box
```

The severity engine converts these detection properties into an overall severity score.

---

# 📐 Severity Calculation

The project assigns a severity weight to each damage category.

| Damage Type        | Class ID | Weight |
| ------------------ | -------: | -----: |
| Longitudinal crack |        0 |   1.00 |
| Transverse crack   |        1 |   1.00 |
| Alligator crack    |        2 |   1.50 |
| Other corruption   |        3 |   0.75 |
| Pothole            |        4 |   2.00 |

The higher the weight, the greater the contribution of that damage type to the overall severity score.

---

# 📏 8. Bounding-Box Area Ratio

For each detected damage:

$$
Area_{\text{bbox}} = (x_2-x_1)(y_2-y_1)
$$

The image area is:

$$
Area_{\text{image}} = W \times H
$$

Therefore:

$$
AreaRatio =
\frac{Area_{\text{bbox}}}
{Area_{\text{image}}}
$$

This measures how much of the image is occupied by the detected damage.

---

# 📈 9. Area Factor

The project converts the area ratio into an area factor:

$$
AreaFactor =
\min(AreaRatio \times 20, 1)
$$

This limits the area contribution to a maximum of 1.

Therefore:

```text
Small bounding box
      ↓
Small area factor
      ↓
Lower severity contribution


Large bounding box
      ↓
Larger area factor
      ↓
Higher severity contribution
```

---

# 🎯 10. Individual Damage Severity

For every detection:

$$
S_i =
W_i \times C_i \times (0.5 + AreaFactor_i)
$$

where:

* $S_i$ = severity contribution of detection $i$
* $W_i$ = damage-class weight
* $C_i$ = YOLO confidence
* $AreaFactor_i$ = normalized bounding-box area factor

The base factor of **0.5** ensures that confidence contributes to the severity even when the detected bounding box is relatively small.

---

# ➕ 11. Total Damage Score

If an image contains multiple detected damages:

$$
D = \sum_{i=1}^{N} S_i
$$

where:

* $N$ = number of detections
* $D$ = total damage score

Therefore, multiple road defects contribute cumulatively to the overall severity.

```text
Detection 1 → Longitudinal crack

Detection 2 → Pothole

Detection 3 → Alligator crack

          │
          ▼
Individual severity scores

          │
          ▼
       Sum scores

          │
          ▼
     Total D score
```

---

# 🔴 12. Severity Index

The project converts the total damage score into a normalized 0–100 severity value:

$$
Severity = \min(15 \times D, 100)
$$

Therefore:

$$
0 \le Severity \le 100
$$

A value closer to 100 represents a higher calculated damage severity.

---

# 🏥 13. Road Health Index — RHI

The Road Health Index is calculated as:

$$
\boxed{RHI = 100 - Severity}
$$

Therefore:

```text
High Severity
      ↓
Low RHI
      ↓
Poorer road condition


Low Severity
      ↓
High RHI
      ↓
Better road condition
```

The RHI ranges from:

$$
0 \le RHI \le 100
$$

---

# 🚦 14. Severity Categories

The project uses the following interpretation:

|   Severity | Condition    |
| ---------: | ------------ |
|     `< 20` | 🟢 Good      |
| `20–39.99` | 🟡 Fair      |
| `40–59.99` | 🟠 Poor      |
| `60–79.99` | 🔴 Very Poor |
|     `≥ 80` | 🚨 Critical  |

> **Note:** These categories are part of the project's derived severity/RHI framework. They are not an official RDD2022 severity standard.

---

# 🧮 15. Complete Severity Example

Suppose the detector identifies:

## Detection 1

```text
Class: Longitudinal crack

Weight = 1.0
Confidence = 0.90
AreaFactor = 0.30
```

Then:

$$
S_1 = 1.0 \times 0.90 \times (0.5+0.30)
$$

$$
S_1 = 0.72
$$

## Detection 2

```text
Class: Pothole

Weight = 2.0
Confidence = 0.85
AreaFactor = 0.50
```

Then:

$$
S_2 = 2.0 \times 0.85 \times (0.5+0.50)
$$

$$
S_2 = 1.70
$$

## Total Damage Score

$$
D = 0.72 + 1.70 = 2.42
$$

## Severity

$$
Severity = \min(15 \times 2.42,100)
$$

$$
Severity = 36.3
$$

## RHI

$$
RHI = 100 - 36.3
$$

$$
\boxed{RHI = 63.7}
$$

The resulting severity falls into the:

🟡 **Fair** category according to the project's thresholds.

---

# 🔗 16. Complete Severity Pipeline

```text
📷 Road Image
      │
      ▼
🧠 YOLOv8n + SCConv + EMA + P2
      │
      ▼
📦 Bounding Boxes
      │
      ├── Damage Class
      ├── Confidence
      └── Bounding Box Area
      │
      ▼
📏 Area Ratio
      │
      ▼
📐 Area Factor
      │
      ▼
⚖️ Class Severity Weight
      │
      ▼
🎯 Individual Severity
      │
      ▼
➕ Total Damage Score
      │
      ▼
🔴 Severity Index
      │
      ▼
🏥 Road Health Index (RHI)
      │
      ▼
🚦 Condition Category
```

---

# 📊 17. Model Training Configuration

The training notebook uses the following configuration:

| Parameter     | Value                       |
| ------------- | --------------------------- |
| Architecture  | YOLOv8n + SCConv + EMA + P2 |
| Dataset       | RDD2022 10K subset          |
| Input size    | `640 × 640`                 |
| Epochs        | 100                         |
| Batch size    | 32                          |
| Device        | CUDA GPU                    |
| AMP           | Disabled                    |
| Workers       | 4                           |
| Cache         | Disabled                    |
| Optimizer     | Auto                        |
| Patience      | 30                          |
| Validation    | Enabled                     |
| Plots         | Enabled                     |
| Seed          | 42                          |
| Deterministic | False                       |

---

# 🗂️ 18. Dataset

The project uses a 10,000-image subset of **RDD2022**.

RDD2022 contains road images collected from multiple countries and includes common road-damage categories such as longitudinal cracks, transverse cracks, alligator cracks, and potholes.

This project uses five classes:

```text
0 → Longitudinal crack
1 → Transverse crack
2 → Alligator crack
3 → Other corruption
4 → Pothole
```

The project dataset is organized into:

```text
RDD2022_CJU_10K/
│
├── train/
│   ├── images/
│   └── labels/
│
├── val/
│   ├── images/
│   └── labels/
│
└── test/
    ├── images/
    └── labels/
```

### YOLO Label Format

```text
class_id x_center y_center width height
```

All coordinates are normalized to the image dimensions.

---

# 🧪 19. Training Pipeline

```text
RDD2022 10K Dataset
        │
        ▼
Dataset Verification
        │
        ▼
YOLO Annotation Validation
        │
        ▼
Dataset YAML Creation
        │
        ▼
Custom SCConv Definition
        │
        ▼
Custom EMA Definition
        │
        ▼
Register Custom Modules
        │
        ▼
Create P2 Model YAML
        │
        ▼
Build YOLOv8n Model
        │
        ▼
Forward-Pass Verification
        │
        ▼
100-Epoch Training
        │
        ▼
Best.pt
        │
        ▼
Validation
        │
        ▼
Final Test Evaluation
        │
        ├── mAP@50
        ├── mAP@50–95
        ├── Precision
        ├── Recall
        ├── Per-class AP
        ├── Confusion Matrix
        ├── PR Curve
        └── F1 Curve
```

---

# 📈 20. Final Model Results

The reported final test results for this project are:

| Metric        |     Result |
| ------------- | ---------: |
| 🎯 mAP@50     | **71.04%** |
| 📊 mAP@50–95  | **42.54%** |
| ✅ Precision   | **75.85%** |
| 🔍 Recall     | **61.20%** |
| ⚙️ Parameters | **4.01 M** |
| 🔥 GFLOPs     |   **15.5** |

The corresponding F1 score can be calculated from precision and recall:

$$
F1 = \frac{2PR}{P+R}
$$

Using:

$$
P = 0.7585,\qquad R = 0.6120
$$

gives approximately:

$$
\boxed{F1 = 67.74\%}
$$

---

# 📚 21. Comparison with Selected Models

| Model                           | mAP@50 (%) | mAP@50–95 (%) | Model Size        |
| ------------------------------- | ---------: | ------------: | ----------------- |
| **YOLOv8n + SCConv + EMA + P2** |  **71.04** |     **42.54** | **4.01 M Params** |
| YOLOv5s                         |      67.20 |         35.50 | 7.03 M Params     |
| YOLOv5l                         |      64.70 |         30.50 | 64.1 M Params     |
| YOLOv7-tiny                     |      64.80 |         31.20 | 6.02 M Params     |
| YOLOv8n                         |      61.80 |         32.30 | 12.9 MB*          |
| YOLOv8 + DAT + GSConv + MPDIoU  |      65.70 |         34.80 | 11.8 MB*          |

> **Note:** The values reported as MB in the final rows come from the corresponding paper's model-size/parameter column and should not automatically be interpreted as the same measurement as the 4.01 M parameter count.

---

# 🧠 22. Why the Architecture Was Modified

## Problem 1 — Small Road Damage

Small cracks and small potholes can be difficult to detect after repeated downsampling.

### Modification

🔍 **P2 Detection Head**

```text
P2 = 160 × 160
Stride = 4
```

This retains more spatial detail.

---

## Problem 2 — Redundant Features

Road scenes contain substantial background information.

### Modification

🧩 **SCConv**

SCConv performs spatial and channel reconstruction at P3 and P4.

---

## Problem 3 — Important Feature Selection

Not every feature location contributes equally to road-damage detection.

### Modification

🎯 **EMA**

EMA is applied throughout the neck to refine fused feature representations.

---

## Problem 4 — Multi-scale Road Damage

Road damage can range from tiny cracks to large potholes.

### Modification

🔄 **P2/P3/P4/P5 Detection**

Four scales allow the detector to process damage at different spatial sizes.

---

# 🏗️ 23. Model Modification Summary

```text
                   BASE YOLOv8n
                       │
          ┌────────────┼────────────┐
          │            │            │
          ▼            ▼            ▼
       Backbone       Neck         Head
          │            │            │
          │            │            │
       SCConv         EMA           P2
          │            │             +
          │            │           P3/P4/P5
          │            │
          └────────────┴─────────────
                       │
                       ▼
              YOLOv8n + SCConv
                    + EMA
                    + P2
```

---

# 💡 24. Main Technical Contribution

The architecture combines three complementary modifications:

$$
\boxed{\text{YOLOv8n + SCConv + EMA + P2}}
$$

### SCConv

Improves the representation of intermediate features.

### EMA

Improves attention and feature refinement after multi-scale fusion.

### P2

Preserves high-resolution information for small damage.

### Bidirectional Fusion

Allows information to flow:

$$
P5 \rightarrow P4 \rightarrow P3 \rightarrow P2
$$

and:

$$
P2 \rightarrow P3 \rightarrow P4 \rightarrow P5
$$

This gives the network both:

* Semantic information
* Fine spatial information

---

# 💻 25. Project Structure

A recommended project structure is:

```text
Road-Damage-Detection/
│
├── 📁 dataset/
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── val/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
│
├── 📁 models/
│   ├── yolov8n_p2_scconv_ema.yaml
│   └── best.pt
│
├── 📁 results/
│   ├── results.png
│   ├── confusion_matrix.png
│   ├── confusion_matrix_normalized.png
│   ├── PR_curve.png
│   └── F1_curve.png
│
├── 📁 inference/
│   ├── predict.py
│   └── severity.py
│
├── 📄 rdd2022_10k.yaml
├── 📓 yolov8n-scconv-ema-p2-10000.ipynb
├── 📄 requirements.txt
└── 📄 README.md
```

---

# ⚙️ 26. Installation

Create a Python environment:

```bash
python -m venv venv
```

Activate it on Linux:

```bash
source venv/bin/activate
```

On Windows:

```powershell
venv\Scripts\activate
```

Install dependencies:

```bash
pip install ultralytics opencv-python pyyaml numpy pandas matplotlib torch torchvision
```

For GPU training, install a PyTorch build compatible with your CUDA environment.

Verify CUDA:

```python
import torch

print("CUDA:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
```

---

# 🏋️ 27. Training

The training configuration used in the project is equivalent to:

```python
model.train(
    data="rdd2022_10k.yaml",
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    amp=False,
    workers=4,
    cache=False,
    patience=30,
    optimizer="auto",
    val=True,
    save=True,
    plots=True,
    seed=42,
    deterministic=False
)
```

The best checkpoint is saved as:

```text
weights/best.pt
```

---

# 🔎 28. Validation

The validation process evaluates:

* mAP@50
* mAP@50–95
* Precision
* Recall

Example:

```python
results = model.val(
    data="rdd2022_10k.yaml",
    split="val",
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    plots=True
)
```

---

# 🧪 29. Final Test Evaluation

The final test evaluation uses:

```python
test_results = model.val(
    data="rdd2022_10k.yaml",
    split="test",
    imgsz=640,
    batch=16,
    device=0,
    workers=4,
    plots=True
)
```

The project generates:

```text
📊 Results
📉 Confusion Matrix
📉 Normalized Confusion Matrix
📈 PR Curve
📈 F1 Curve
📊 Per-class AP
```

---

# 🎯 30. Inference

After training, the best checkpoint can be loaded using:

```python
from ultralytics import YOLO

model = YOLO("best.pt")

results = model.predict(
    source="road_image.jpg",
    imgsz=640,
    conf=0.25
)
```

The prediction provides:

```text
Bounding boxes
Class labels
Confidence scores
```

These values can then be passed into the severity engine.

---

# 🏥 31. Detection → Severity → RHI

The final application pipeline is:

```text
                  📷 IMAGE
                     │
                     ▼
        YOLOv8n + SCConv + EMA + P2
                     │
                     ▼
             🔎 DETECTION
                     │
        ┌────────────┼─────────────┐
        ▼            ▼             ▼
      Class       Confidence      Box
        │            │             │
        └────────────┼─────────────┘
                     ▼
              📐 Area Ratio
                     │
                     ▼
              ⚖️ Severity Weight
                     │
                     ▼
             Individual Score
                     │
                     ▼
            Total Damage Score
                     │
                     ▼
             🔴 Severity Index
                     │
                     ▼
          🏥 Road Health Index
                     │
                     ▼
           🚦 Condition Level
```

---

# 📌 32. Important Interpretation

The severity and RHI component should be interpreted as a **project-defined analytical layer on top of object detection**.

YOLO itself predicts the damage bounding boxes and classes.

The project then derives severity using:

$$
\boxed{
S_i = W_i C_i(0.5 + AreaFactor_i)
}
$$

$$
\boxed{
D = \sum_i S_i
}
$$

$$
\boxed{
Severity = \min(15D,100)
}
$$

and:

$$
\boxed{
RHI = 100 - Severity
}
$$

This means the system is not claiming that YOLO directly learns an official pavement-condition severity label.

---

# 🧪 33. Testing the Custom Modules

The notebook performs a forward-pass test for both SCConv and EMA.

For example:

```python
import torch

x = torch.randn(
    1,
    64,
    80,
    80
)

scconv = ScConv(64)

with torch.no_grad():
    y = scconv(x)

assert y.shape == x.shape
```

EMA is tested similarly:

```python
ema = EMA(64)

with torch.no_grad():
    y2 = ema(x)

assert y2.shape == x.shape
```

This verifies that the custom modules preserve the expected tensor dimensions.

---

# 🛠️ 34. Troubleshooting

## `Can't get attribute 'ScConv'`

If a saved model contains custom modules, the Python process must know about the custom class definitions.

Register:

```python
import ultralytics.nn.tasks as tasks

tasks.ScConv = ScConv
tasks.EMA = EMA
```

before loading the checkpoint.

---

## CUDA Unavailable

Check:

```python
import torch

print(torch.cuda.is_available())

print(
    torch.cuda.get_device_name(0)
    if torch.cuda.is_available()
    else "CPU"
)
```

---

## Dataset YAML Error

Verify:

```text
train/images
train/labels

val/images
val/labels

test/images
test/labels
```

and ensure the class IDs are in the range:

```text
0–4
```

---

# 🔐 35. Reproducibility

The training notebook uses:

```text
Seed = 42
```

However:

```text
deterministic = False
```

Therefore, exact bit-for-bit reproduction is not guaranteed across different environments, GPUs, CUDA versions, PyTorch versions, or library versions.

For stronger reproducibility, record:

```text
Python version
PyTorch version
Ultralytics version
CUDA version
GPU model
Dataset version
Training configuration
Model YAML
best.pt
```

---

# 📚 36. Related Research

The architecture was developed in the context of road-damage detection research using RDD2022.

Two relevant comparison papers used in the project are:

## YOLOv8-PD

Zeng & Zhong, **"YOLOv8-PD: an improved road damage detection algorithm based on YOLOv8n"**, *Scientific Reports*, 2024.

The architecture in that work introduces components including:

* BOT
* LSKA
* C2fGhost
* LSCD-Head

and reports experiments on RDD2022.

## Enhanced YOLOv8

Jiang, **"Road damage detection and classification using deep neural networks"**, *Discover Applied Sciences*, 2024.

The model combines:

* Deformable Attention Transformer
* GSConv/Slim-Neck
* MPDIoU

for road-damage detection.

---

# 📊 37. Model Comparison Summary

The reported project result is:

$$
mAP@50 = 71.04\%
$$

and:

$$
mAP@50-95 = 42.54\%
$$

These values can be compared with published results, but comparisons should account for differences in:

* Dataset split
* Class definitions
* Preprocessing
* Training configuration
* Implementation
* Hardware
* Evaluation procedure

Therefore, published-paper values should not automatically be interpreted as controlled head-to-head experiments.

---

# 🌍 38. Potential Applications

The system can be adapted for:

* 🚗 Vehicle-mounted road inspection
* 🚁 Drone-based road inspection
* 🛣️ Highway monitoring
* 🏙️ Smart-city infrastructure monitoring
* 🏗️ Road maintenance planning
* 📊 Road-condition analytics
* 🚧 Damage-priority assessment
* 🗺️ GIS-based road-condition mapping

---

# 🔮 39. Future Improvements

## 📹 Video-Based Detection

Run the detector on road inspection videos and track damage across consecutive frames.

## 🗺️ GPS/GIS Integration

Attach detected damage and RHI values to geographic coordinates.

## 📱 Mobile/Edge Deployment

Export the model to an optimized inference format for embedded devices.

## 📐 Better Geometric Severity

Use:

* Crack length
* Crack width
* Damage area
* Road-lane area
* Perspective correction

rather than bounding-box area alone.

## 🧠 Dedicated Severity Model

Train a separate severity classifier/regressor using manually verified severity labels.

## 📊 Temporal Road-Health Monitoring

Track RHI over time:

```text
Year 1 → RHI 86
Year 2 → RHI 73
Year 3 → RHI 58
Year 4 → RHI 41
```

This could provide a road deterioration trend.

---

# ⚠️ 40. Limitations

The current project has several limitations:

1. The severity index is **derived**, rather than directly learned from human-rated severity labels.
2. Bounding-box area is only an approximation of actual damage area.
3. A bounding box around a thin crack can contain substantial undamaged road pixels.
4. Confidence scores represent detector confidence, not physical damage severity.
5. The class severity weights are project-defined.
6. The RHI is a project-defined index and is not an official RDD2022 or government pavement-condition index.
7. Published-model comparisons are affected by differences in experimental setup.
8. Real-world deployment would require testing across weather, lighting, camera, and road-surface conditions.

---

# 🏁 41. Final Summary

The project proposes:

$$
\boxed{\textbf{YOLOv8n + SCConv + EMA + P2}}
$$

as a customized architecture for road-damage detection.

The three major modifications have complementary roles:

```text
             YOLOv8n
                │
       ┌────────┼────────┐
       ▼        ▼        ▼
    SCConv     EMA      P2
       │        │        │
       ▼        ▼        ▼
  Better      Better    Better
  feature     feature   small-object
  refinement  attention detection
       │        │        │
       └────────┼────────┘
                ▼
       P2/P3/P4/P5 Detection
                │
                ▼
         Road Damage Results
                │
                ▼
         Severity Estimation
                │
                ▼
         Road Health Index
```

## Final Reported Performance

| Metric        |     Result |
| ------------- | ---------: |
| 🎯 mAP@50     | **71.04%** |
| 📈 mAP@50–95  | **42.54%** |
| 🎯 Precision  | **75.85%** |
| 🔍 Recall     | **61.20%** |
| 📊 F1         | **67.74%** |
| ⚙️ Parameters | **4.01 M** |
| 🔥 GFLOPs     |   **15.5** |

The project combines:

> **Deep learning + multi-scale object detection + attention + feature reconstruction + small-object detection + derived severity analysis**

to create an end-to-end road-condition analysis pipeline. 🚧🛣️🤖📊🏥

---

# ⭐ Project Pipeline at a Glance

```text
📷 INPUT ROAD IMAGE
        ↓
🧠 YOLOv8n
        ↓
🧩 SCConv
        ↓
🎯 EMA
        ↓
🔍 P2 + P3 + P4 + P5
        ↓
📦 DAMAGE DETECTION
        ↓
🏷️ DAMAGE CLASSIFICATION
        ↓
📐 BOUNDING-BOX ANALYSIS
        ↓
⚖️ SEVERITY SCORE
        ↓
🔴 SEVERITY INDEX
        ↓
🏥 ROAD HEALTH INDEX
        ↓
🚦 ROAD CONDITION
```

---

# ❤️ Acknowledgement

This project builds upon the YOLO family of object detectors and the RDD2022 road-damage dataset. The custom architecture and severity/RHI layer are developed as part of this project.

<p align="center">
<b>Built for intelligent road infrastructure monitoring. 🛣️🤖🚧</b>
</p>

<p align="center">
<b>🚧 Detect • Analyze • Assess • Maintain 🛣️</b>
</p>
