import numpy as np
from mlp import NeuralNetwork, test_nn

def convert_to_serializable(obj):
    """Recursively convert numpy types to Python standard types."""
    if isinstance(obj, np.ndarray):
        # Apply conversion to all elements of the array
        return obj.astype(np.int64).tolist()
    elif isinstance(obj, np.int64):
        return int(obj)
    elif isinstance(obj, dict):
        return {convert_to_serializable(k): convert_to_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    return obj
# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[1, 0, 0], [0, 1, 0], [0, 1, 0], [1, 0, 0]])

# Define activations: layer -> activation function
activations = [
    (2, "relu"),  # Input layer (2 neurons, activation relu)
    (4, "tanh"),  # Hidden layer (4 neurons, activation relu)
    (6, "tanh"),  # Hidden layer (6 neurons, activation tanh)
    (6, "tanh"),  # Hidden layer (6 neurons, activation tanh)
    (4, "tanh"),  # Hidden layer (4 neurons, activation tanh)
    (3, "sigmoid") # Output layer (3 neurons, activation output, softmax or sigmoid)
]

# Initialize NeuralNetwork with specified activations
nn = NeuralNetwork(activations, learning_rate=0.001)

# Train the model
nn.train(X, y, epochs=20000, timing=True)

# Test the model
predictions = nn.predict(X)
print("Raw Predictions (probabilities):")
print(predictions)

# Convert predictions to class labels based on the max probability (i.e., argmax for one-hot encoding)
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted class labels:")
print(predicted_labels)

# Convert the true values to single class labels (same as for predictions)
y_true_labels = np.argmax(y, axis=1)

# Now, use these labels to evaluate the performance
tester = test_nn(convert_to_serializable(y_true_labels), convert_to_serializable(predicted_labels))

print("\nClassification report:")
tester.print_report()

print("\nConfusion matrix:")
tester.print_confusion_matrix()
