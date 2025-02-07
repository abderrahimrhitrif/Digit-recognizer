# CNN MNIST Classifier from Scratch

## Overview

This project implements a Convolutional Neural Network (CNN) from scratch to classify handwritten digits from the MNIST dataset. The goal is to take an image, process it, and predict the corresponding digit using a neural network. The final model achieves **95.3% accuracy** after **10,000 iterations**.

## Dataset

The MNIST dataset consists of 28x28 grayscale images of handwritten digits (0-9). Each image is represented as a **784-dimensional vector** (flattened from 28x28 pixels).

## Network Architecture

The neural network consists of **three layers**:

- **Input Layer (Layer 0):** 784 nodes (one per pixel)
- **Hidden Layer (Layer 1):** 10 nodes
- **Hidden Layer (Layer 2):** Additional hidden layer added to improve accuracy, 10 nodes
- **Output Layer (Layer 3):** 10 nodes (corresponding to digit classes 0-9)

## Forward Propagation

### Mathematical Representation:

1. **Input Layer:**  
   `A^(0) = X`  
   (Where X is the input image vector of shape **(784, 1)**)
   
2. **Hidden Layer 1:**  
   `Z^(1) = W^(1) * A^(0) + b^(1)`  
   `A^(1) = g(Z^(1))` (using **ReLU activation function**)  
   
3. **Hidden Layer 2:**  
   `Z^(2) = W^(2) * A^(1) + b^(2)`  
   `A^(2) = g(Z^(2))` (using **ReLU activation function**)  

4. **Output Layer:**  
   `Z^(3) = W^(3) * A^(2) + b^(3)`  
   `A^(3) = softmax(Z^(3))` (converting outputs to probability distribution)  

## Activation Functions

- **ReLU (Rectified Linear Unit)** is used in the hidden layers:
  `ReLU(x) = max(0, x)`
- **Softmax** is applied to the final layer to convert outputs into probabilities.

## Backpropagation

To train the network, backpropagation is used to update weights and biases by minimizing the error between predictions and actual labels:

1. Compute the loss between predicted output and actual label.
2. Calculate gradients of the loss with respect to weights and biases.
3. Update weights and biases using gradient descent.

## Training Process

- **Optimizer:** Gradient Descent
- **Number of Iterations:** 10,000
- **Final Accuracy:** **95.3%**

## How to Run

1. Clone this repository:
   ```bash
   git clone https://github.com/abderrahimrhitrif/Digit-recognizer.git
   ```
2. Install dependencies:
   ```bash
   pip install pandas numpy matplotlib
   ```
3. Open and run the Jupyter Notebook:
   ```bash
   jupyter notebook digit-recongnizer.ipynb
   ```

## Results

The final model achieves **95.3% accuracy** after 10,000 iterations.

## Future Improvements

- Experiment with different activation functions.
- Implement more hidden layers and tune hyperparameters.
- Explore advanced optimizers such as Adam or RMSprop.

## License

This project is open-source under the MIT License.

---

For any queries, feel free to reach out or open an issue in the repository!

