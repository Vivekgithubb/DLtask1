```markdown
# Deep Learning

This project implements three fundamental neural network architectures **entirely from scratch using Python and NumPy**, without relying on high-level deep learning frameworks such as TensorFlow or PyTorch.

The goal of this project is to build a transparent understanding of deep learning by manually implementing:

- Forward propagation
- Backpropagation
- Gradient computation
- Weight initialization
- Optimization algorithms
- Energy-based modeling (RBM)

All models are trained and evaluated on the **MNIST handwritten digit dataset**.

---

## Implemented Models

### 1️⃣ Multi-Layer Perceptron (MLP)

- Architecture: `784 → 128 → 10`
- Activation Functions:
  - ReLU (hidden layer)
  - Softmax (output layer)
- Loss Function: Cross-Entropy
- Optimizer: Stochastic Gradient Descent (SGD)
- Task: 10-class handwritten digit classification

---

### 2️⃣ Sparse Autoencoder

- Architecture: `784 → 64 → 784`
- Activation Functions:
  - ReLU (encoder)
  - Sigmoid (decoder)
- Loss Function: Mean Squared Error (MSE)
- Regularization: L1 sparsity penalty on latent activations
- Tasks:
  - Dimensionality reduction
  - Data reconstruction
  - Anomaly detection via reconstruction error

---

### 3️⃣ Restricted Boltzmann Machine (RBM)

- Visible Units: 784
- Hidden Units: 64
- Training Algorithm: Contrastive Divergence (CD-1)
- Tasks:
  - Generative feature learning
  - Visualization of learned filters
  - Modeling joint probability distribution of input data

---

## Repository Structure
```

.
├── models.py # Contains MLP, Autoencoder, and RBM implementations
├── train.py # Main training and execution script
├── MLPTrainLossVsAccuracy.png # MLP training loss & accuracy curves
├── SparseEncoderOrgvsrecon.png # Autoencoder reconstruction results
├── RBMLearnerFilters.png # Learned RBM filters visualization
├── outlier_detection.png # Anomaly detection histogram

````

---

## Installation

Make sure you have the following installed:

- Python 3.x
- NumPy
- Matplotlib
- Scikit-learn

Install dependencies using:

```bash
pip install numpy matplotlib scikit-learn
````

---

## How to Run

From the project directory, execute:

```bash
python train.py
```

This script will:

- Load and preprocess the MNIST dataset
- Train the Multi-Layer Perceptron
- Train the Sparse Autoencoder
- Train the Restricted Boltzmann Machine
- Generate and save output visualizations

---

## Results Overview

This implementation demonstrates:

- Stable convergence of MLP with high classification accuracy
- Effective compressed latent representations using sparse autoencoding
- Successful anomaly detection via reconstruction error thresholding
- Meaningful parts-based feature learning using RBM filters

---

## Key Learning Outcomes

- Manual implementation of backpropagation
- Understanding of He and Xavier weight initialization
- Application of L1 regularization for sparsity
- Implementation of Contrastive Divergence for generative models
- Clear distinction between discriminative and generative learning approaches

---

## Important Note

This project intentionally avoids the use of:

- PyTorch
- TensorFlow
- Keras
- Automatic differentiation libraries

All gradient computations and optimization steps are implemented manually using NumPy.

---

## Author

**D Vivek Pai**
USN: NNM23IS038
Course: IS1103-1

---

## License

This project is developed strictly for academic and educational purposes.

```

---

If you want, I can also generate:

- A more impressive portfolio-style README (with badges and images embedded)
- A minimal clean academic README
- A resume-optimized GitHub project description
- A version with embedded output images

Just tell me the vibe you want: **assignment clean** or **GitHub portfolio strong** 🚀
```
