# Image Classification on CIFAR-10 with Residual Networks

## Abstract
This paper studies how deep residual networks perform on natural colour images and whether heavy data augmentation is necessary.

## Dataset
We use the CIFAR-10 dataset with 50,000 training images and 10,000 test images across ten object classes such as airplane, ship and frog. Images are 32 by 32 pixels in colour.

## Methodology
We train a ResNet-18 model with stochastic gradient descent, a learning rate of 0.1 that decays by a factor of ten every 30 epochs, a batch size of 128 and 90 epochs. Data augmentation uses random crops and horizontal flips.

## Results
The ResNet-18 model achieved 92.4 percent test accuracy. Removing data augmentation reduced accuracy to 87.1 percent, so augmentation proved important.

## Limitations
Training required a high-end GPU for about 6 hours. The images are small and the class set is limited, so conclusions may not hold for larger photographs.

## Conclusion
Residual networks with data augmentation give strong accuracy on CIFAR-10, but the compute cost is much higher than for simple digit recognition.
