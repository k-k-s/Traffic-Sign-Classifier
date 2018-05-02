## Project: Build a Traffic Sign Recognition Program
[![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)

Objective
---
Train a Deep Neural Network to identify German Traffic Signs.

The Project
---
The goals / steps of this project are the following:
* Load the data set
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


Data set Summary and Exploration:

We can observe that there are unequal number of examples of different classes. This can be a problem.
I ignore this. 

Preprocessing:

I converted the RGB image to gray scale to reduce training time. I then normalized between 0.1 and 0.9 as seen in the lecture, to make it suitable for optimization. Something to do with well-conditioned system.

Neural Net Architecture:
 
After several iterations and experiments, I landed with the following architecture and parameters.

Input:  a 32x32x1 image after preprocess
Layer 1: Convolutional. The output shape is 28x28x6.
Activation. RELU
Pooling. The output shape is 14x14x6.
Layer 2: Convolutional. The output shape is 10x10x16.
Activation. RELU
Pooling. The output shape should be 5x5x16.
Flatten the output.
Layer 3: Convolutional. Output shape is 1x1x400
Flatten the output.
Combine these  into a 800 1D vector.
Add Dropout.
Layer 4: Fully Connected. Output is 400
Activation. RELU
Add Dropout.
Layer 5: Fully Connected. Output is 125.
Activation. RELU
Add Dropout.
Output : Layer 5: Fully Connected (Logits). 43 outputs.

Parameters:

Epochs = 90
Batch size = 150
Learning rate = 0.0008
Mu =0
Sigma = 0.1
Keep_prob = 0.7
Optimizer  = Adam

Process behind the selection:

I initially tried LeNet architecture but got only 89 percent Validation accuracy. So I tried changing the hyper parameters, still no luck. Then I read the paper by LeCun on traffic sign classification and adapted the model.
I iterated to find the optimal batch size of 150, (learning rate,keep_prob and epoch) parameter set.

Results:
Validation accuracy: 95.7 % after 90 epochs.
Test accuracy: 93.4 % 

Further improvement is possible in preprocessing by generating new images of classes which have less than the mean number of examples per class. Also, the Neural Net architecture can be explored in terms of convolutional layers, inception, regularization, etc.

Testing on New Images, Difficulties:

I took 5 images from google images.  These images after preprocessing appear distorted. This can be a problem.

The model predicted a double curve as a children crossing.
A 60 Km speed sign was predicted to be a keep right sign.
A Stop sign was predicted to be a roundabout mandatory.
A No Entry sign was correctly predicted,
A 100 km speed sign was predicted to be a No Entry sign.

This bad performance is due to loss of resolution when resizing the original image as can be seen in the output. We have to solve the resizing problem.

Softmax Probabilities:

As can be seen in the output, the one it predicted correctly has probability of 1.  The probabilities for the other classes in the prediction are not close. More testing is required to judge the fidelity of the model.


Improvements Suggested:
---
Histogram equalization for adjusting image intensities to enhance contrast. 
https://docs.opencv.org/3.1.0/d5/daf/tutorial_py_histogram_equalization.html
Preprocessing options http://cs231n.github.io/neural-networks-2/#datapre

Dropout analysis  https://pgaleone.eu/deep-learning/regularization/2017/01/10/anaysis-of-dropout/

Batch normalization to make model robust to bad initialization :http://cs231n.github.io/neural-networks-2/#batchnorm

Use ELU (speed optimized) : https://arxiv.org/abs/1511.07289
Dying RELUs - use Leaky RELUs https://datascience.stackexchange.com/questions/5706/what-is-the-dying-relu-problem-in-neural-networks

Visualizing the architecture - use Tensor Board : https://www.tensorflow.org/programmers_guide/graph_viz

Other Strategies to tune the parameters : http://cs231n.github.io/neural-networks-3/#hyper
Cyclical Learning rate : https://arxiv.org/abs/1506.01186

Image Augmentation : https://medium.com/@vivek.yadav/dealing-with-unbalanced-data-generating-additional-data-by-jittering-the-original-image-7497fe2119c3

Try Transfer Learning to train the model on test image examples and change the weights accordingly.

### Dependencies
This lab requires:

* [CarND Term1 Starter Kit](https://github.com/udacity/CarND-Term1-Starter-Kit)

The lab environment can be created with CarND Term1 Starter Kit. Click [here](https://github.com/udacity/CarND-Term1-Starter-Kit/blob/master/README.md) for the details.


