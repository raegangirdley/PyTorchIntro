# PyTorchIntro

Created simple skeleton code for training a neural network on images through PyTorch. A random tensor is given, so no images are actually being trained on in this file.

Has an ImageReader class that flattens a given multi-dimensional input into a one-dimensional vector. Creates container of layers that take said image and output 10 labels that the model believes it is trying to predict. Accounts for non-linearity.

Does a forward pass through layers and activation functions. Finds index in given data with highest probability of matching desired label and prints prediction of what it believes matches label the most.
