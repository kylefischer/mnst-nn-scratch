# train_mnist.py
import numpy as np
import os
import certifi
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Fix for SSL certificate verification on macOS
os.environ["SSL_CERT_FILE"] = certifi.where()

from nn import Neural_Network

# You need TensorFlow installed for this import:
# pip install tensorflow
from tensorflow.keras.datasets import mnist


def load_mnist(normalize: bool = True):
    """
    Load MNIST and return flattened train/test splits.
    """
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    # X_* are (N, 28, 28) -> flatten to (N, 784)
    X_train = X_train.reshape(-1, 28 * 28).astype(np.float32)
    X_test = X_test.reshape(-1, 28 * 28).astype(np.float32)

    if normalize:
        X_train /= 255.0
        X_test /= 255.0

    return X_train, y_train, X_test, y_test


def main():
    # 1. Load data
    X_train, y_train, X_test, y_test = load_mnist(normalize=True)
    print("Train shape:", X_train.shape, y_train.shape)
    print("Test shape:", X_test.shape, y_test.shape)

    # 2. Initialize model
    model = Neural_Network(
        input_size=784,   # 28x28
        hidden_size=128,  # you can change this
        output_size=10    # digits 0-9
    )

    # 3. Train
    learning_rate = 0.1
    iterations = 1000
    batch_size = 128

    print("Starting training...")
    history = model.train(
        X_train,
        y_train,
        X_val=X_test,
        y_val=y_test,
        learning_rate=learning_rate,
        iterations=iterations,
        batch_size=batch_size,
        print_every=50,
    )

    # 4. Evaluate
    print("Evaluating on test set...")
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    print(f"Test accuracy: {accuracy * 100:.2f}%")

    # Optional: inspect a few predictions
    for i in range(5):
        print(f"Example {i}: predicted={y_pred[i]}, true={y_test[i]}")

    # 5. Plot metrics
    iterations_range = history["iterations"]
    
    # Plot Loss
    plt.figure(figsize=(10, 5))
    plt.plot(iterations_range, history["train_loss"], label="Train Loss")
    plt.plot(iterations_range, history["val_loss"], label="Validation Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/loss_curve.png")
    print("Saved plots/loss_curve.png")

    # Plot Accuracy
    plt.figure(figsize=(10, 5))
    plt.plot(iterations_range, history["val_acc"], label="Validation Accuracy", color="orange")
    plt.xlabel("Iterations")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.legend()
    plt.grid(True)
    plt.savefig("plots/accuracy_curve.png")
    print("Saved plots/accuracy_curve.png")


if __name__ == "__main__":
    main()
