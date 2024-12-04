import numpy as np
from mlp import NeuralNetwork, test_nn


# XOR input and output
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([[0], [1], [1], [0]])

# Define activations: layer -> activation function
activations = [
    (2, "relu"),  # Input layer (2 neurons, activation relu)
    (4, "tanh"),  # Hidden layer (4 neuron, activation tanh)
    (1, "output") # Output layer (1 neurons, activation output (sigmoid as default))
]
# Initialize NeuralNetwork with specified activations
nn = NeuralNetwork(activations, learning_rate=0.01)

# Train the model
nn.train(X, y, epochs=1000000, timing=True)


# Test the model
predictions = nn.predict(X)
print(predictions)
predictions = (predictions > 0.5).astype(int)
print("Predictions:")
print(predictions)  # Threshold predictions

tester = test_nn(y, predictions)

print("\nClasification report:")
tester.print_report()
print("\nConfusion matrix:")
tester.print_confusion_matrix()
