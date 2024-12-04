Here’s a README file for the neural network implementation:

---

# Neural Network Implementation

This repository contains a basic implementation of a neural network from scratch using NumPy, with forward and backward propagation, loss calculation, and performance metrics such as confusion matrix and classification report.

## Features

- **Multi-layer Neural Network:** Configurable number of layers with various activation functions (sigmoid, relu, tanh).
- **Training Process:** Implements backpropagation with gradient descent to minimize binary cross-entropy loss.
- **Performance Metrics:** Includes methods to compute confusion matrix, classification report (precision, recall, F1-score), and accuracy.
- **Timing Support:** Option to time the training process and display elapsed time per epoch.
- **JSON and Tabular Output:** The confusion matrix and classification report can be printed as a formatted table or returned as JSON.

## Requirements

- Python 3.x
- NumPy
- Tabulate (for formatted printing)

To install the required dependencies, run:

```bash
pip install numpy tabulate
```

## How It Works

### `NeuralNetwork` Class

This class implements a feedforward neural network with the following key methods:

- `__init__(self, activations, learning_rate=0.001)`: Initializes the network with given layer structure and learning rate.
  - **`activations`**: A list of tuples representing the number of neurons and activation function for each layer. Example: `[(2, "relu"), (4, "tanh"), (1, "output")]`
  - **`learning_rate`**: The learning rate for gradient descent.

- `initialize_weights()`: Initializes weights and biases for each layer using Xavier initialization.

- `forward_propagation(X)`: Performs forward propagation to compute the output of the network for a given input `X`.

- `backpropagation(X, y)`: Performs backpropagation to adjust the weights based on the difference between predicted and actual values.

- `train(X, y, epochs=10000, timing=False)`: Trains the network for a specified number of epochs and prints loss at each epoch. Optionally, it can time the training.

- `predict(X)`: Predicts the output for a given input `X`.

- `binary_crossentropy(y_true, y_pred)`: Computes the binary cross-entropy loss between true and predicted labels.

- `format_elapsed_time(elapsed)`: Static method to format elapsed time into a human-readable format.

### `test_nn` Class

This class provides various methods to evaluate the performance of the neural network by calculating the confusion matrix, classification report, and accuracy:

- `__init__(self, y_true, y_pred)`: Initializes the class with true and predicted labels.
  
- `confusion_matrix()`: Computes the confusion matrix for multi-class classification.

- `print_confusion_matrix()`: Pretty-prints the confusion matrix as a table.

- `confusion_matrix_json()`: Returns the confusion matrix as a JSON string.

- `classification_report()`: Computes precision, recall, F1-score, and accuracy for each class.

- `report_json()`: Returns the classification report as a JSON string.

- `print_report()`: Pretty-prints the classification report.

## Example Usage

### Creating and Training the Neural Network

```python
# Example data (X: input features, y: labels)
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])  # Example XOR inputs
y = np.array([[0], [1], [1], [0]])  # Example XOR outputs

# Define the network architecture: [(#neurons, "activation")]
activations = [(2, "relu"), (4, "tanh"), (1, "output")]

# Create a Neural Network instance
nn = NeuralNetwork(activations, learning_rate=0.01)

# Train the neural network
nn.train(X, y, epochs=10000, timing=True)

# Make predictions
predictions = nn.predict(X)
print(predictions)
```

### Evaluating Performance with `test_nn`

```python
# Create an instance of the test_nn class with true and predicted values
true_labels = [0, 1, 1, 0]
predicted_labels = [0, 1, 0, 1]

# Initialize the test_nn class
evaluator = test_nn(true_labels, predicted_labels)

# Print confusion matrix
evaluator.print_confusion_matrix()

# Print classification report
evaluator.print_report()
```

## License

This code is licensed under the GNU General Public License v2. - see the [LICENSE](LICENSE) file for details.

---

This README provides a comprehensive guide to understand the purpose, features, and usage of the neural network code you’ve written. It also includes an example to show how to use the neural network class and evaluate its performance with the `test_nn` class.
