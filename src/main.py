import math
import random

import matplotlib.pyplot as plt


def sigmoid(x):
    return 1 / (1 + math.exp(-x))


def tanh(x):
    return math.tanh(x)


def relu(x):
    return max(0, x)


def main():
    random.seed(42)

    x1 = 3
    x2 = 2
    y_actual = 17  # y = 3 * x1 + 4 * x2

    w1 = random.uniform(1, 10)
    w2 = random.uniform(1, 10)

    learning_rate = 0.01
    epochs = 20

    errors = []
    y_predictions = []

    print("Activation examples")
    print(f"sigmoid(1): {sigmoid(1):.4f}")
    print(f"tanh(1):    {tanh(1):.4f}")
    print(f"relu(-1):   {relu(-1):.4f}")
    print()

    print(f"Initial weights: w1 = {w1:.4f}, w2 = {w2:.4f}")

    for epoch in range(epochs):
        y_pred = x1 * w1 + x2 * w2
        error = (y_actual - y_pred) ** 2

        errors.append(error)
        y_predictions.append(y_pred)

        d_loss_d_prediction = -2 * (y_actual - y_pred)
        grad_w1 = d_loss_d_prediction * x1
        grad_w2 = d_loss_d_prediction * x2

        w1 -= learning_rate * grad_w1
        w2 -= learning_rate * grad_w2

        print(
            f"Epoch {epoch + 1:02d}: "
            f"y_pred = {y_pred:.4f}, "
            f"error = {error:.6f}, "
            f"w1 = {w1:.4f}, "
            f"w2 = {w2:.4f}"
        )

    print(f"\nFinal weights: w1 = {w1:.4f}, w2 = {w2:.4f}")
    print(f"Final prediction: {y_predictions[-1]:.4f}")
    print(f"Actual value:     {y_actual:.4f}")

    plt.figure()
    plt.plot(range(1, epochs + 1), errors, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Squared Error")
    plt.title("Error Progression Across Epochs")
    plt.grid(True)
    plt.show()

    plt.figure()
    plt.plot(range(1, epochs + 1), y_predictions, marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Predicted y")
    plt.title("Prediction Progression Across Epochs")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()
