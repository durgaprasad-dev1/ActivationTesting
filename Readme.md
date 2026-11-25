```markdown
# Comparing Activations on Non-Linear Data

This project demonstrates how different neural network activation functions (**Sin**, **Tanh**, **ReLU**, and **SwiGLU**) perform when learning highly non-linear decision boundaries. By training parameter-efficient models on synthetic spiral datasets, we compare their convergence rates, final classification losses, and qualitative decision boundaries.

## 📌 Table of Contents
- [Overview](#overview)
- [Activation Functions Tested](#activation-functions-tested)
- [Dataset](#dataset)
- [Model Architectures](#model-architectures)
  - [Individual Activation Model](#individual-activation-model)
  - [Unified Multi-Activation Model](#unified-multi-activation-model)
- [Key Experiments](#key-experiments)
- [Requirements](#requirements)
- [Results & Key Takeaways](#results--key-takeaways)

---

## 🔍 Overview
Traditional activation functions like **ReLU** and **Tanh** are standard baselines, but custom periodic activations like **Sin** or gated linear units like **SwiGLU** (widely used in modern LLMs) can offer superior performance or unique properties when modeling complex structures. This codebase provides a standardized benchmark to visualize how these activations behave on a 2D double-spiral classification task.

## ⚡ Activation Functions Tested
1. **ReLU (Rectified Linear Unit)**: The standard baseline ($f(x) = \max(0, x)$).
2. **Tanh (Hyperbolic Tangent)**: A smooth, zero-centered sigmoidal curve.
3. **Sin (Sine Activation)**: A periodic activation function ($f(x) = \sin(x)$) capable of learning high-frequency coordinate patterns.
4. **SwiGLU (Swish Gated Linear Unit)**: A gating mechanism that splits projections and applies Swish gating ($f(x) = \text{Swish}(W_1 x) \otimes W_2 x$).

## 📊 Dataset
We generate a synthetic **Double Spiral Dataset** featuring two highly interleaved, non-linear spiral classes in a 2D coordinate space:
- **Standard Dataset**: 4,000 total points (2,000 points per class).
- **Large Dataset**: 20,000 total points (10,000 points per class) to test representation scalability.

## 🏗️ Model Architectures
### Individual Activation Model
- **Type**: Multi-Layer Perceptron (MLP)
- **Parameters**: ~15,000 parameters
- **Structure**: 
  `Input (2D) -> Hidden Layer (120) -> Activation -> Hidden Layer (120) -> Activation -> Output (1D)`

### Unified Multi-Activation Model
- **Type**: Parallel Multi-Branch Neural Network
- **Parameters**: ~5,200 parameters (extremely lightweight)
- **Structure**: Runs four parallel sub-branches (ReLU, Tanh, Sin, SwiGLU) with 30 hidden dimensions each. Features from all four branches are concatenated and passed to a final linear classification layer.

## 🧪 Key Experiments
1. **Standard Benchmark**: Train individual models for 150 epochs on the 4,000-point dataset and compare decision boundaries.
2. **Scaling Benchmark**: Train individual models on the larger 20,000-point dataset using adjusted batch sizes.
3. **Unified Modeling**: Train the parallel multi-branch model to test if combining multiple activation behaviors leads to better representation capability with fewer parameters.

## 💻 Requirements
To run the notebook, install the required dependencies:
```bash
pip install numpy matplotlib torch
```

## 📈 Results & Key Takeaways
- **Periodic vs. Linear**: Periodic activations like **Sin** are highly effective at forming smooth, wrapping spiral boundaries, while **ReLU** constructs piecewise-linear boundaries.
- **SwiGLU Scalability**: **SwiGLU** offers excellent representation capability because of its gating mechanisms, though it requires parameter projection splits.
- **Unified Advantage**: The **Unified Multi-Activation Model** combines the unique decision behaviors of all activations, allowing it to solve the complex spiral boundary with significantly fewer total parameters (~5k parameters) compared to standard individual networks.
```