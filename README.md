# 🧠 Neural Network From Scratch

> A ground-up implementation of neural network fundamentals — feed-forward prediction, squared error loss, gradient descent, and weight updates, all written without frameworks.

---

## 📌 Project Overview

This lab strips away all framework abstractions to show exactly how a neural network learns. Using a tiny two-input regression example, it demonstrates how predictions change as weights are updated step by step via gradient descent — making the internals of deep learning fully transparent.

The goal is conceptual clarity: before using TensorFlow or Keras, this project shows what actually happens under the hood.

---

## 🔍 What This Lab Demonstrates

| Concept | Implementation |
|---|---|
| Feed-forward prediction | Manual dot product: `x1*w1 + x2*w2` |
| Loss function | Squared error: `(prediction - target)²` |
| Gradient calculation | Analytical gradient per weight |
| Weight update rule | `w = w - lr * gradient` |
| Activation functions | Sigmoid, Tanh, ReLU — all implemented manually |
| Visualization | Error and prediction plotted across epochs |

---

## ⚙️ How It Works

The target value is defined as:

```text
y = 3 * x1 + 4 * x2
```

The script initializes two random weights, computes a prediction, calculates squared error, applies gradient descent, and plots how error and prediction evolve over iterations — making each step of learning visible.

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python 3.x | Core implementation language |
| NumPy | Numerical operations |
| Matplotlib | Training curve visualization |

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/sainathac/neural-network-from-scratch.git
cd neural-network-from-scratch
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the lab
```bash
python src/main.py
```

---

## 📚 Part of the Deep Learning Lab Series

This is **Lab 1 of 6** in a TensorFlow/Keras deep learning learning series:

1. **[Neural Network From Scratch](https://github.com/sainathac/neural-network-from-scratch)** ← You are here
2. [MNIST Digit Classifier](https://github.com/sainathac/mnist-digit-classifier)
3. [Fashion-MNIST Model Lifecycle](https://github.com/sainathac/fashion-mnist-model-lifecycle)
4. [Cats vs Dogs CNN Classifier](https://github.com/sainathac/cats-vs-dogs-cnn-classifier)
5. [VGG16 Transfer Learning Classifier](https://github.com/sainathac/vgg16-transfer-learning-classifier)
6. [IMDB Sentiment RNN](https://github.com/sainathac/imdb-sentiment-rnn)

---

## 👤 Author

**Sainath AC**
AI Automation & RPA Engineer | Data Science Practitioner

[![GitHub](https://img.shields.io/badge/GitHub-sainathac-181717?logo=github)](https://github.com/sainathac)
