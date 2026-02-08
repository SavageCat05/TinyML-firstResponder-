# Emergency Intent Detection Model v2 Specification

## Overview
This document outlines the architectural changes from v1 (standard CNN) to v2 (MobileNet-style) to address robustness against noise, transient sounds, and muffled audio while maintaining TinyML efficiency.

## Core Architectural Changes

| Feature | v1 Implementation | v2 Implementation | Benefit |
| :--- | :--- | :--- | :--- |
| **Noise Handling** | Standard Conv2D | **Squeeze-and-Excitation (SE)** | Dynamically recalibrates channel importance to suppress noise and amplify emergency signatures. |
| **Transient Detection** | Global Average Pooling | **Global Max Pooling** | Captures peak energy events (screams, gunshots) that would be washed out by averaging. |
| **Feature Extraction** | Standard Conv layers | **Inverted Residual Blocks** | Expands features into higher dimensions to find hidden patterns before compressing, helping with muffled audio. |
| **Efficiency** | Standard Conv2D | **Depthwise Separable Convs** | Reduces computational cost (MACs) and parameter count, allowing for deeper networks within the same size budget. |

## Detailed Architecture

### Input
- **Shape:** `(40, 50, 1)` (Log-mel Spectrogram: 40 mels, 50 time steps, 1 channel)

### Backbone (MobileNet-style)
1.  **Stem:**
    -   Conv2D (16 filters, 3x3, stride 2) -> BN -> ReLU
    -   *Purpose:* Rapid spatial downsampling and initial feature extraction.

2.  **Bottleneck Blocks (Inverted Residuals):**
    -   **Block 1:** 16 filters, Expansion=1, Stride=1 (Depthwise + Pointwise)
    -   **Block 2:** 24 filters, Expansion=4, Stride=2 (Expand -> Depthwise -> Pointwise)
    -   **Block 3:** 24 filters, Expansion=4, Stride=1 + **SE Block**
    -   **Block 4:** 40 filters, Expansion=4, Stride=2 + **SE Block**
    -   **Block 5:** 40 filters, Expansion=4, Stride=1 + **SE Block**
    -   *Note:* Residual connections added where input/output shapes match.

### Head
1.  **Post-Conv:** 1x1 Conv to expand to high-dim feature space (e.g., 128 filters).
2.  **Pooling:** **Global Max Pooling** (Critical for short bursts).
3.  **Classifier:**
    -   Dense (Dropout 0.2)
    -   Dense (Num Classes, Softmax)

## Constraints
-   **Model Size:** < 500 KB (Int8 Quantized)
-   **Latency:** < 100 ms on ESP32
-   **Framework:** TensorFlow/Keras (Project Standard)
