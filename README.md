# Deep-Learning-Implementations-Exploring-Neural-Architectures
# Deep Learning Architectures: From Perceptrons to ResNet

This repository features implementations of deep learning architectures, from perceptrons to advanced CNNs like AlexNet, VGG-16, GoogleNet, and ResNet. Built using TensorFlow, Keras, and PyTorch, these models demonstrate key design principles, training, and evaluation. A valuable resource for learners and practitioners exploring neural networks and deep learning.

## 📂 Repository Structure

### **Fundamental Neural Networks**
- **`Perceptron_with_Linearity.ipynb`** - Implements a basic perceptron model with a linear activation function for binary classification.
- **`Multilayered_neural_network_with_Linearity.ipynb`** - A simple feedforward neural network demonstrating how multiple layers enhance learning capacity.

### **Classic Convolutional Networks**
- **`LENET-5.ipynb`** - Implementation of LeNet-5, one of the earliest CNN architectures designed for handwritten digit recognition.
- **`ALEXNET.ipynb`** - Implements AlexNet, which introduced ReLU activation, dropout, and overlapping pooling, winning the 2012 ImageNet challenge.
- **`VGG-16.ipynb`** - A deep CNN with 16 layers using small 3x3 convolutions, known for its simplicity and strong image classification performance.

### **Advanced Convolutional Architectures**
- **`GoogleNet.ipynb`** - Implements GoogleNet (Inception Network), which uses inception modules to enhance efficiency and accuracy while reducing computational cost.
- **`ResNet.ipynb`** - Implementation of ResNet, which introduces residual learning with skip connections, allowing deeper networks without vanishing gradients.
- **`CNN.ipynb`** - A general convolutional neural network (CNN) implementation using TensorFlow, trained on the CIFAR-10 dataset.

### **Feature Maps and Visualization**
- **`Convolutional_Feature_Maps_on_7_7_Input_Data_with_Keras.ipynb`** - Demonstrates feature map visualization for a CNN using a 7×7 input image, illustrating how convolutional layers transform inputs.
- **`Convolutional_Feature_Maps_on_8_8_Input_Data_with_Keras.ipynb`** - Similar to the 7×7 feature map visualization, but applied to 8×8 input data, showing how filters extract spatial features.

### Usage
Each Jupyter Notebook file (`.ipynb`) can be run independently. Open them using Jupyter Notebook or Google Colab to explore model architectures and training procedures.

## 📚 References
- **Deep Learning with Python** – François Chollet  
- **CS231n: Convolutional Neural Networks for Visual Recognition** – Stanford  
- **PyTorch & TensorFlow Official Documentation**
