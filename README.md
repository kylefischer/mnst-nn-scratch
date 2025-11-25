# MNIST Neural Network

A simple, fully-connected neural network implementation from scratch using NumPy to classify MNIST digits.

## Features

*   **From Scratch Implementation**: The core neural network logic (forward pass, backpropagation, activation functions) is implemented in pure NumPy without using high-level deep learning frameworks for the model itself.
*   **Custom Training Loop**: Includes a flexible training loop with mini-batch gradient descent.
*   **Performance Metrics**: Tracks and logs training loss, validation loss, and validation accuracy.
*   **Visualization**: Automatically generates plots for loss and accuracy curves.

## Prerequisites

*   Python 3.x
*   NumPy
*   TensorFlow (only for loading the MNIST dataset)
*   Matplotlib (for plotting)

## Installation

1.  Clone the repository.
2.  Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the training script:

```bash
python3 train_mnst.py
```

This will:
1.  Load the MNIST dataset.
2.  Initialize the neural network.
3.  Train the model for 1000 iterations.
4.  Evaluate the model on the test set.
5.  Save performance plots to the `plots/` directory.

## File Structure

*   `nn.py`: Contains the `Neural_Network` class and helper functions (ReLU, Softmax, etc.).
*   `train_mnst.py`: Main script to load data, train the model, and generate plots.
*   `requirements.txt`: List of Python dependencies.
*   `plots/`: Directory where generated plots are saved.

## Results

After training, you can check the `plots/` directory for:
*   `loss_curve.png`: Training and validation loss over time.
*   `accuracy_curve.png`: Validation accuracy over time.

