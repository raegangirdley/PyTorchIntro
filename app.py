import os #provides way to interact with operating system
import torch
from torch import nn #imports PyTorch's neural network module
from torch.utils.data import DataLoader #provides data loading utilities, such as for loading datsets
from torchvision import datasets, transforms #used for loading and transforming image datasets

#Class defined as subclass of nn.Module
class ImageReader(nn.Module):
    #Constructor
    def __init__(self):
        super().__init__()
        #Turns multi-dimensional input, like an image,
        #and converts to one dimensional vector
        self.flatten = nn.Flatten()
        
        #Creates container of layers that data will flow through
        self.linear_relu_stack = nn.Sequential(
            #Fully connected layer, takes 28*28 image and
            #outputs to 512-dimensional feature vector
            nn.Linear(28*28, 512),
            #Non-linear activation function, introduces non-linearity, outputs node.
            #If node is important. Otherwise, outputs 0.
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            #Outputs 10 labels model is trying to predict.
            nn.Linear(512, 10),
        )

    #Method that describes flow of data.
    def forward(self, x):
        x = self.flatten(x)
        #Passes x through defined sequence of linear layers
        #and activation functions. Logits represents raw
        #scores for each 10 classes.
        logits = self.linear_relu_stack(x)
        return logits

#Sends instance of ImageReader to "cuda". Can also be sent to
#cpu with '.to"cpu"'.
model = ImageReader().to("cuda")
#some_data is a random tensor for input
some_data = torch.randn(1, 28, 28).to("cuda")
logits = model(some_data)
#Converts to probabilities
pred_probab = nn.Softmax(dim=1)(logits)
#Finds index with highest probability
y_pred = pred_probab.argmax(1)

for row in some_data:
    for value in row:
        print(value)

#Prints prediction
print(f"Prediction: {y_pred}")

#This class's prediction is determined by the output size
#of the 10 final layers given. The model is being given
#a random tensor in this class, so technically it is not
#predicting anything and is, essentially, "guessing".

#With actual images, it could be given classes, or "labels",
#and learn the patterns correctly so that the model could
#accurately predict if something matched the correct class 
#or not.

#TL;DR, this is skeleton code for training a neural network
#on images.