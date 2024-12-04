import numpy as np
import time
import json
from tabulate import tabulate


class NeuralNetwork:
    def __init__(self, activations, learning_rate=0.001):
        """
        Initializes the neural network with given activation layers and learning rate.
        
        Parameters:
        activations (list of tuples): Each tuple contains (#neurons, "activation") for each layer.
                                       Example: [(2, "relu"), (4, "tanh"), (6, "tanh"), (4, "tanh"), (1, "output")]
        learning_rate (float): The learning rate for weight updates. Default is 0.001.
        """
        self.activations = activations
        self.learning_rate = learning_rate
        self.layers = [layer[0] for layer in activations]  # Extract number of neurons
        self.activation_types = [layer[1] for layer in activations]  # Extract activation names
        self.weights, self.biases = self.initialize_weights()
        self.activation_functions = {
            "sigmoid": (self.sigmoid, self.sigmoid_derivative),
            "relu": (self.relu, self.relu_derivative),
            "tanh": (self.tanh, self.tanh_derivative),
            "softmax": (self.softmax, self.softmax_derivative),
            "output": (self.softmax, self.softmax_derivative)  # Use softmax for the output layer in multi-class classification
        }

    def sigmoid(self, x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        """Derivative of the sigmoid activation function."""
        return x * (1 - x)

    def relu(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)

    def relu_derivative(self, x):
        """Derivative of the ReLU activation function."""
        return np.where(x > 0, 1, 0)

    def tanh(self, x):
        """Tanh activation function."""
        return np.tanh(x)

    def tanh_derivative(self, x):
        """Derivative of the tanh activation function."""
        return 1 - np.tanh(x) ** 2

    def softmax(self, x):
        """Softmax activation function."""
        exp_values = np.exp(x - np.max(x, axis=-1, keepdims=True))  # Stability trick to prevent overflow
        return exp_values / np.sum(exp_values, axis=-1, keepdims=True)

    def softmax_derivative(self, output, true_labels):
        """
        Computes the derivative of the softmax function in a manner appropriate for backpropagation.

        Parameters:
        output (ndarray): The output of the softmax function (predicted probabilities).
        true_labels (ndarray): The true labels (one-hot encoded).

        Returns:
        ndarray: The gradient of the softmax output with respect to the loss.
        """
        # Gradient for softmax in multi-class classification is:
        # delta = softmax(output) - true_labels
        return output - true_labels


    def categorical_crossentropy(self, y_true, y_pred):
        """
        Categorical Cross-Entropy Loss function.
        
        Parameters:
        y_true (ndarray): True labels (one-hot encoded).
        y_pred (ndarray): Predicted probabilities.

        Returns:
        float: The computed categorical cross-entropy loss.
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # Prevent log(0)
        return -np.mean(np.sum(y_true * np.log(y_pred), axis=-1))

    def initialize_weights(self):
        """
        Initializes weights and biases for each layer in the network using the Xavier initialization.

        Returns:
        tuple: A tuple containing weights and biases for all layers.
        """
        weights = []
        biases = []
        for i in range(len(self.layers) - 1):
            input_size = self.layers[i]
            output_size = self.layers[i + 1]
            limit = np.sqrt(6 / (input_size + output_size))  # Xavier initialization limit
            weights.append(np.random.uniform(-limit, limit, (input_size, output_size)))
            biases.append(np.zeros((1, output_size)))
        return weights, biases

    def forward_propagation(self, X):
        """
        Performs forward propagation through the network.

        Parameters:
        X (ndarray): Input data.

        Returns:
        list: A list of activations for each layer.
        """
        activations = [X]
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            activation_func_name = self.activation_types[i + 1]
            func, _ = self.activation_functions[activation_func_name]
            activations.append(func(z))
        return activations
    def backpropagation(self, X, y):
        """
        Performs backpropagation to compute gradients and update weights and biases.

        Parameters:
        X (ndarray): Input data.
        y (ndarray): True labels (one-hot encoded).
        """
        m = X.shape[0]  # Number of training examples
        activations = self.forward_propagation(X)

        # Output layer error and delta
        output_error = y - activations[-1]
        delta = self.softmax_derivative(activations[-1], y)  # Use both activations and y
        deltas = [delta]

        # Backpropagate for hidden layers
        for i in range(len(self.weights) - 2, -1, -1):
            _, derivative_func = self.activation_functions[self.activation_types[i + 1]]
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * derivative_func(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases using gradient descent
        for i in range(len(self.weights)):
            gradient_w = np.dot(activations[i].T, deltas[i]) / m
            gradient_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] += self.learning_rate * gradient_w
            self.biases[i] += self.learning_rate * gradient_b

    def train(self, X, y, epochs=10000, timing=False):
        """
        Trains the neural network for a given number of epochs.

        Parameters:
        X (ndarray): Input data.
        y (ndarray): True labels.
        epochs (int): Number of training epochs. Default is 10000.
        timing (bool): If True, prints the time taken for each epoch.
        """
        total_time = 0
        for epoch in range(epochs):
            if timing:
                start_time_epoch = time.perf_counter()  # High precision timer
            self.backpropagation(X, y)
            output = self.forward_propagation(X)[-1]
            current_loss = self.categorical_crossentropy(y, output)
            if timing:
                end_time_epoch = time.perf_counter()  # High precision timer
                elapsed_time_epoch = end_time_epoch - start_time_epoch
                print(f"Epoch {epoch}, Loss: {current_loss}, Time: {self.format_elapsed_time(elapsed_time_epoch)}")
                total_time += elapsed_time_epoch
            else:
                print(f"Epoch {epoch}, Loss: {current_loss}")
        if timing:
            print(f"Total time: {self.format_elapsed_time(total_time)}")

    def predict(self, X):
        """
        Predicts the output for the given input data.

        Parameters:
        X (ndarray): Input data.

        Returns:
        ndarray: The predicted output.
        """
        return self.forward_propagation(X)[-1]

    @staticmethod
    def format_elapsed_time(elapsed):
        """
        Formats the elapsed time in a human-readable format.

        Parameters:
        elapsed (float): The elapsed time in seconds.

        Returns:
        str: The formatted elapsed time.
        """
        if elapsed >= 3600:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = elapsed % 60
            return f"{hours} hours, {minutes} minutes, {seconds:.2f} seconds"
        elif elapsed >= 60:
            minutes = int(elapsed // 60)
            seconds = elapsed % 60
            return f"{minutes} minutes, {seconds:.2f} seconds"
        elif elapsed >= 1:
            return f"{elapsed:.4f} seconds"
        elif elapsed >= 1e-3:
            ms = elapsed * 1e3
            return f"{ms:.3f} ms"
        elif elapsed >= 1e-6:
            μs = elapsed * 1e6
            return f"{μs:.3f} µs"
        else:
            ns = elapsed * 1e9
            return f"{ns:.3f} ns"

class test_nn:
    def __init__(self, y_true, y_pred):
        self.y_true = np.array(y_true).flatten()  # Ensure both are 1D arrays
        self.y_pred = np.array(y_pred).flatten()
        self.validate_inputs()
        self.classes = np.unique(self.y_true)  # Get unique classes in y_true

    def validate_inputs(self):
        """Ensure y_true and y_pred have the same length."""
        if len(self.y_true) != len(self.y_pred):
            raise ValueError("y_true and y_pred must have the same length.")

    def confusion_matrix(self):
        """Calculate confusion matrix for multi-class classification."""
        cm = np.zeros((len(self.classes), len(self.classes)), dtype=int)

        # Populate the confusion matrix
        for true, pred in zip(self.y_true, self.y_pred):
            true_index = np.where(self.classes == true)[0][0]
            pred_index = np.where(self.classes == pred)[0][0]
            cm[true_index, pred_index] += 1

        # Convert matrix to a JSON-like structure with true and predicted labels
        cm_dict = {}
        for i, true_label in enumerate(self.classes):
            cm_dict[true_label] = {}
            for j, pred_label in enumerate(self.classes):
                cm_dict[true_label][pred_label] = cm[i, j]

        return cm_dict

    def print_confusion_matrix(self):
        """Pretty print confusion matrix as a table."""
        cm_dict = self.confusion_matrix()
        # Prepare table headers and rows
        headers = ['True \ Pred'] + list(self.classes)
        rows = []

        for true_label in self.classes:
            row = [true_label]  # Start with the true label
            for pred_label in self.classes:
                row.append(cm_dict[true_label].get(pred_label, 0))
            rows.append(row)

        # Print the table using tabulate
        print(tabulate(rows, headers=headers, tablefmt='grid'))

    def confusion_matrix_json(self):
        """Return confusion matrix as a JSON string."""
        cm_dict = self.confusion_matrix()
        return json.dumps(cm_dict, indent=4)

    def classification_report(self):
        """Calculate precision, recall, F1-score, and accuracy for each class."""
        cm = self.confusion_matrix()
        report = {}

        # Iterate through the confusion matrix dictionary
        for true_label, row in cm.items():
            # True positives: correct predictions for the class
            tp = row[true_label]
            # False positives: total predictions minus true positives
            fp = sum(row.values()) - tp
            fn = sum(cm[other_label][true_label] for other_label in cm) - tp  # False negatives
            tn = sum(sum(row.values()) for row in cm.values()) - tp - fp - fn  # True negatives

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
            accuracy = (tp + tn) / sum(sum(row.values()) for row in cm.values()) if sum(sum(row.values()) for row in cm.values()) > 0 else 0.0

            report[true_label] = {
                "Precision": precision,
                "Recall": recall,
                "F1-Score": f1_score,
                "Accuracy": accuracy
            }

        # Calculate overall accuracy
        overall_accuracy = sum(cm[label][label] for label in cm) / sum(sum(row.values()) for row in cm.values())
        report["Accuracy"] = overall_accuracy

        return report

    def report_json(self):
        """Return the classification report as a JSON string with proper types."""
        report = self.classification_report()
        # Convert NumPy types to native Python types
        report_with_correct_types = self.convert_types(report)
        return json.dumps(report_with_correct_types, indent=4)

    def print_report(self):
        # Parse the JSON string into a dictionary
        try:
            report = json.loads(self.report_json())
            for class_label, metrics in report.items():
                if class_label != "Accuracy":  # Skip the overall accuracy field
                    print(f"Class {class_label}:")
                    for metric, value in metrics.items():
                        print(f"  {metric}: {value * 100}%")
            # Print overall accuracy
            print(f"\nOverall Accuracy: {report.get('Accuracy', 'N/A') * 100}%")
        except json.JSONDecodeError:
            print("Invalid JSON format")

    def convert_types(self, obj):
        if isinstance(obj, dict):
            return {self.convert_types(k): self.convert_types(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_types(i) for i in obj]
        elif isinstance(obj, np.int32):
            return int(obj)
        elif isinstance(obj, np.float32):
            return float(obj)
        else:
            return obj