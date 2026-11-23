# Handwritten Digit Recognition with Convolutional Networks

## Abstract
The objective of this research is to evaluate whether a small convolutional neural network can recognise handwritten digits more accurately than a classical support vector machine baseline while remaining cheap to train.

## Dataset
All experiments use the MNIST dataset, which contains 60,000 training images and 10,000 test images of handwritten digits from 0 to 9. Each image is 28 by 28 pixels in grayscale. We held out 5,000 training images for validation.

## Methodology
The first method is a convolutional neural network with three convolutional layers followed by two fully connected layers. It was trained with the Adam optimizer using a learning rate of 0.001, a batch size of 64 and 20 epochs. The second method is a support vector machine with an RBF kernel, trained on flattened pixel values and used as the baseline.

## Results
The convolutional network reached a test accuracy of 99.2 percent, while the support vector machine baseline reached 97.8 percent. Training the network took about 12 minutes on a single consumer GPU.

## Limitations
The study only covers clean grayscale digits. Robustness to noise, rotation and different handwriting styles was not evaluated, and the findings may not transfer to colour images.

## Conclusion
A compact convolutional network clearly outperforms the support vector machine baseline on MNIST and is practical to train on modest hardware. Future work will test noisy and colour datasets.
