# Neural Network From Scratch

A simple from-scratch neural network fundamentals lab demonstrating feed-forward prediction, squared error, gradient calculation, and manual weight updates.

## Overview

This project uses a tiny two-input regression example to show how a model prediction changes as weights are updated with gradient descent. It is intentionally small so the learning loop is easy to inspect.

## What This Lab Demonstrates

- Manual feed-forward prediction
- Squared error loss
- Gradient calculation for each weight
- Weight updates using a learning rate
- Basic activation functions: sigmoid, tanh, and ReLU
- Error and prediction visualization across epochs

## How It Works

The target value is generated from:

```text
y = 3 * x1 + 4 * x2
```

The script initializes two random weights, predicts a value using `x1 * w1 + x2 * w2`, computes squared error, applies gradient descent, and plots how the prediction and error evolve.

## Run

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the lab:

```bash
python src/main.py
```

## Notes

This is an educational fundamentals lab, not a production machine learning model. It is useful for understanding the intuition behind gradient descent before moving into TensorFlow/Keras.
