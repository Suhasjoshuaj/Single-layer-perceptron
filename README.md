Single-Layer Perceptron for MNIST Classification

Overview
This project implements a **Single-Layer Perceptron (SLP)** using **TensorFlow/Keras** to classify handwritten digits from the **MNIST dataset**. The model consists of a single-layer neural network trained using a simple learning algorithm.

Features
- Uses the **MNIST dataset** (28x28 grayscale images of digits 0-9).
- Implements a **Single-Layer Perceptron** using TensorFlow/Keras.
- Trains the model on **handwritten digit classification**.
- Visualizes training performance and predictions.

Prerequisites
Ensure you have the following dependencies installed:

```bash
pip install numpy matplotlib tensorflow
```

Installation
Clone this repository to your local machine:

```bash
git clone https://github.com/yourusername/single-layer-perceptron-mnist.git
cd single-layer-perceptron-mnist
```

Dataset
The model uses the **MNIST dataset**, which is automatically downloaded via Keras:

```python
from tensorflow import keras
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
```

Model Architecture
- **Input Layer:** 784 neurons (flattened 28x28 images)
- **Output Layer:** 10 neurons (one for each digit 0-9)
- **Activation Function:** Softmax

Training the Model
Run the Jupyter Notebook or Python script to train the model:

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Load dataset
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()

# Normalize and flatten images
X_train = X_train.reshape(-1, 28*28) / 255.0
X_test = X_test.reshape(-1, 28*28) / 255.0

# Define the perceptron model
model = keras.models.Sequential([
    keras.layers.Dense(10, activation='softmax', input_shape=(784,))
])

# Compile and train
model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
```

Evaluating the Model
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc:.4f}")
```

Visualizing Predictions
```python
import matplotlib.pyplot as plt
predictions = model.predict(X_test)
plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
plt.title(f"Predicted Label: {np.argmax(predictions[0])}")
plt.show()
```

## Limitations
- This is a simple **single-layer** model; deeper networks perform better.
- Lacks hidden layers, making it **less powerful** for complex patterns.
- Can be improved with **multi-layer perceptrons (MLPs)**.

## Next Steps
- Implement a **Multi-Layer Perceptron (MLP)** with hidden layers.
- Experiment with different **optimizers and activation functions**.
- Apply to **more complex datasets**.



