# MNIST_Resnet18
This program utilizes the ResNet-18 deep learning structure to train MNIST dataset consisting of 60000 handwritten digits of 0~9.

To begin training the data, open TrainingMNIST.m, and run the program. 
It will also give you the plot for confusion matrix and some randomized classification example.
You can also modify the epoch and iteration inside the file.
With the default parameter, it takes about 16 minutes to train the data (single GPU: RTX 2060 Super).

So, if you do not want to train the data, you can directly use the AfterTraining.m and run, to obtain the confusion matrix and samples.


Thank you for the MNIST dataset for MATLAB:
https://www.mathworks.com/matlabcentral/fileexchange/69480-sample-deep-network-training-with-mnist-and-cifar by Masayuki Tanaka (2020)
and the dataset initial package:
http://yann.lecun.com/exdb/mnist/ by Yann LeCun et.al
