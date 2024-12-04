import numpy as np
import time
import json
from tabulate import tabulate
class NeuralNetwork:
    def __init__(self, activations, learning_rate=0.001):
        """
        activations: List of tuples [(#neurons, "activation")]
        Example: [(2, "relu"), (4, "tanh"), (6, "tanh"), (4, "tanh"), (1, "output")]
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
            "output": (self.sigmoid, self.sigmoid_derivative)  # Default for output
        }

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def relu(self, x):
        return np.maximum(0, x)

    def relu_derivative(self, x):
        return np.where(x > 0, 1, 0)

    def tanh(self, x):
        return np.tanh(x)

    def tanh_derivative(self, x):
        return 1 - np.tanh(x) ** 2

    def binary_crossentropy(self, y_true, y_pred):
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def initialize_weights(self):
        weights = []
        biases = []
        for i in range(len(self.layers) - 1):
            input_size = self.layers[i]    # Current layer size
            output_size = self.layers[i + 1]  # Next layer size
            limit = np.sqrt(6 / (input_size + output_size))
            weights.append(np.random.uniform(-limit, limit, (input_size, output_size)))
            biases.append(np.zeros((1, output_size)))
        return weights, biases

    def forward_propagation(self, X):
        activations = [X]  # The activations list starts with the input
        for i in range(len(self.weights)):
            z = np.dot(activations[-1], self.weights[i]) + self.biases[i]
            # Get activation function for current layer
            activation_func_name = self.activation_types[i + 1]
            func, _ = self.activation_functions[activation_func_name]
            activations.append(func(z))  # Apply activation function to z
        return activations

    def backpropagation(self, X, y):
        m = X.shape[0]
        activations = self.forward_propagation(X)

        # Calculate the output error and delta for output layer
        output_error = y - activations[-1]
        _, output_derivative = self.activation_functions[self.activation_types[-1]]
        delta = output_error * output_derivative(activations[-1])
        deltas = [delta]

        # Backpropagate for each hidden layer
        for i in range(len(self.weights) - 2, -1, -1):
            _, derivative_func = self.activation_functions[self.activation_types[i + 1]]
            delta = np.dot(deltas[-1], self.weights[i + 1].T) * \
                derivative_func(activations[i + 1])
            deltas.append(delta)
        deltas.reverse()

        # Update weights and biases
        for i in range(len(self.weights)):
            gradient_w = np.dot(activations[i].T, deltas[i]) / m
            gradient_b = np.sum(deltas[i], axis=0, keepdims=True) / m
            self.weights[i] += self.learning_rate * gradient_w
            self.biases[i] += self.learning_rate * gradient_b

    def train(self, X, y, epochs=10000, timing=False):
        total_time = 0
        for epoch in range(epochs):
            if timing:
                start_time_epoch = time.perf_counter()  # Use perf_counter for higher precision
            self.backpropagation(X, y)
            if True:
                output = self.forward_propagation(X)[-1]
                current_loss = self.binary_crossentropy(y, output)
                if timing:
                    end_time_epoch = time.perf_counter()  # Use perf_counter for higher precision
                    elapsed_time_epoch = end_time_epoch - start_time_epoch
                    print(f"Epoch {epoch}, Loss: {current_loss}, Time: {self.format_elapsed_time(elapsed_time_epoch)}")
                    total_time += elapsed_time_epoch
                else:
                    print(f"Epoch {epoch}, Loss: {current_loss}")
        if timing:
            print(f"Total time: {self.format_elapsed_time(total_time)}")

    def predict(self, X):
        return self.forward_propagation(X)[-1]

    @staticmethod
    def format_elapsed_time(elapsed):
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