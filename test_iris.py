import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
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

# Load Iris dataset (use pandas to read the CSV)
df = pd.read_csv('iris.csv')

# Separate features and labels
X = df.iloc[:, 1:5].values  # Features (SepalLengthCm, SepalWidthCm, PetalLengthCm, PetalWidthCm)
y = df.iloc[:, 5].values  # Target (Species)

# Encode target labels (Species) into integers
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# Convert y_encoded to one-hot encoding
y_one_hot = np.zeros((y_encoded.size, y_encoded.max() + 1))
y_one_hot[np.arange(y_encoded.size), y_encoded] = 1

# Define activations: layer -> activation function
activations = [
    (4, "relu"),  # Input layer (4 neurons, activation relu) - 4 features in the dataset
    (8, "relu"),  # Hidden layer (8 neurons, activation relu)
    (6, "relu"),  # Hidden layer (6 neurons, activation relu)
    (6, "relu"),  # Hidden layer (6 neurons, activation relu)
    (6, "relu"),  # Hidden layer (6 neurons, activation relu)
    (6, "relu"),  # Hidden layer (6 neurons, activation relu)
    (6, "relu"),  # Hidden layer (6 neurons, activation relu)
    (3, "output") # Output layer (3 neurons, activation output, softmax for multi-class)
]

# Initialize NeuralNetwork with specified activations
nn = NeuralNetwork(activations, learning_rate=0.001)

# Train the model
nn.train(X, y_one_hot, epochs=100000, timing=True)

# Test the model
predictions = nn.predict(X)
print("Raw Predictions (probabilities):")
print(predictions)

# Convert predictions to class labels based on the max probability (i.e., argmax for one-hot encoding)
predicted_labels = np.argmax(predictions, axis=1)
print("Predicted class labels:")
print(predicted_labels)

# Convert the true values to single class labels (same as for predictions)
y_true_labels = np.argmax(y_one_hot, axis=1)

# Now, use these labels to evaluate the performance
tester = test_nn(convert_to_serializable(y_true_labels), convert_to_serializable(predicted_labels))

print("\nClassification report:")
tester.print_report()

print("\nConfusion matrix:")
tester.print_confusion_matrix()
