# nn.py
import numpy as np


def relu(x: np.ndarray) -> np.ndarray:
    """ReLU activation."""
    return np.maximum(0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of ReLU."""
    return (x > 0).astype(float)


def softmax(z: np.ndarray) -> np.ndarray:
    """
    Softmax along the last axis (classes).
    z: (N, C)
    """
    # numerical stability
    z = z - np.max(z, axis=1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)


def to_one_hot(y: np.ndarray, num_classes: int = 10) -> np.ndarray:
    """
    Convert integer labels (N,) into one-hot encodings (N, num_classes).
    """
    N = y.shape[0]
    one_hot = np.zeros((N, num_classes), dtype=np.float32)
    one_hot[np.arange(N), y] = 1.0
    return one_hot


class Neural_Network:
    """
    Simple fully-connected neural network for MNIST:
    - Input: 784 (28x28 flattened)
    - Hidden: configurable (default 128) with ReLU
    - Output: 10 classes with softmax
    """

    def __init__(self, input_size: int = 784, hidden_size: int = 128, output_size: int = 10):
        self.inputLayerSize = input_size
        self.hiddenLayerSize = hidden_size
        self.outputLayerSize = output_size

        # He initialization for ReLU layer
        self.W1 = np.random.randn(self.inputLayerSize, self.hiddenLayerSize) * np.sqrt(
            2.0 / self.inputLayerSize
        )
        self.b1 = np.zeros((1, self.hiddenLayerSize), dtype=np.float32)

        self.W2 = np.random.randn(self.hiddenLayerSize, self.outputLayerSize) * np.sqrt(
            2.0 / self.hiddenLayerSize
        )
        self.b2 = np.zeros((1, self.outputLayerSize), dtype=np.float32)

    def forward(self, X: np.ndarray) -> np.ndarray:
        """
        Forward pass.
        X: (N, input_size)
        Returns:
            yHat: (N, output_size) softmax probabilities
        """
        self.z1 = np.dot(X, self.W1) + self.b1        # (N, hidden)
        self.a1 = relu(self.z1)                       # (N, hidden)
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # (N, output)
        self.yHat = softmax(self.z2)                  # (N, output)
        return self.yHat

    def costFunction(self, X: np.ndarray, y_onehot: np.ndarray) -> float:
        """
        Cross-entropy loss over the batch.
        X: (N, input_size)
        y_onehot: (N, output_size)
        """
        yHat = self.forward(X)
        N = X.shape[0]
        # add epsilon to avoid log(0)
        eps = 1e-8
        loss = -np.sum(y_onehot * np.log(yHat + eps)) / N
        return float(loss)

    def costFunctionPrime(self, X: np.ndarray, y_onehot: np.ndarray):
        """
        Backpropagation: compute gradients of the loss w.r.t. W1, b1, W2, b2.
        """
        N = X.shape[0]

        # Forward pass (also caches z1, a1, z2)
        yHat = self.forward(X)

        # dL/dz2 for softmax + cross-entropy
        delta2 = (yHat - y_onehot) / N               # (N, output)
        dJdW2 = np.dot(self.a1.T, delta2)            # (hidden, output)
        dJdb2 = np.sum(delta2, axis=0, keepdims=True)

        # Backprop into hidden layer
        delta1 = np.dot(delta2, self.W2.T) * relu_grad(self.z1)  # (N, hidden)
        dJdW1 = np.dot(X.T, delta1)                              # (input, hidden)
        dJdb1 = np.sum(delta1, axis=0, keepdims=True)

        return dJdW1, dJdb1, dJdW2, dJdb2

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None,
        learning_rate: float = 0.01,
        iterations: int = 1000,
        batch_size: int = 64,
        print_every: int = 50,
    ):
        """
        Train the network using mini-batch gradient descent.

        X: (N, input_size), pixel values in [0, 1]
        y: (N,), integer labels 0-9
        X_val: (M, input_size), optional validation data
        y_val: (M,), optional validation labels
        """
        N = X.shape[0]
        y_onehot_full = to_one_hot(y, num_classes=self.outputLayerSize)

        history = {
            "train_loss": [],
            "val_loss": [],
            "val_acc": [],
            "iterations": []
        }

        if X_val is not None and y_val is not None:
             y_val_onehot = to_one_hot(y_val, num_classes=self.outputLayerSize)

        for i in range(iterations):
            # sample a mini-batch (without replacement)
            idx = np.random.choice(N, batch_size, replace=False)
            X_batch = X[idx]
            y_batch = y_onehot_full[idx]

            dW1, db1, dW2, db2 = self.costFunctionPrime(X_batch, y_batch)

            # Gradient descent updates
            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

            if (i + 1) % print_every == 0 or i == 0:
                cost = self.costFunction(X, y_onehot_full)
                history["train_loss"].append(cost)
                history["iterations"].append(i + 1)
                
                log_msg = f"Iteration {i+1}/{iterations}, loss = {cost:.4f}"

                if X_val is not None and y_val is not None:
                    val_cost = self.costFunction(X_val, y_val_onehot)
                    val_pred = self.predict(X_val)
                    val_acc = np.mean(val_pred == y_val)
                    
                    history["val_loss"].append(val_cost)
                    history["val_acc"].append(val_acc)
                    log_msg += f", val_loss = {val_cost:.4f}, val_acc = {val_acc*100:.2f}%"
                
                print(log_msg)

        return history

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class labels for X.
        Returns:
            (N,) integer labels.
        """
        probs = self.forward(X)
        return np.argmax(probs, axis=1)
