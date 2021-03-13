import os
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# Check to see if hardware accelerator is available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'using "{device}" device')


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.ReLU()
        )

    # Never call this method directly
    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


# Create instance of NeuralNetwork, move it to the device and print its structure
model = NeuralNetwork().to(device)
print(model)

X = torch.rand(1, 28, 28, device=device)
logits = model(X)
pred_probab = nn.Softmax(dim=1)(logits)
y_pred = pred_probab.argmax(1)
print(f"Predicted class: {y_pred}")


# Model Layers
input_image = torch.rand(3, 28, 28)
print(input_image.size())

# Flatten converts each 28x28 2D image into contiguous array of 784 pixel values
flatten = nn.Flatten()
flat_image = flatten(input_image)
print(flat_image.size())

# Linear layer applies a linear transformation on the input using its stored weights and biases
layer1 = nn.Linear(in_features=28*28, out_features=20)
hidden1 = layer1(flat_image)
print(hidden1.size())

# ReLU layer introduces non-linearity
print(f"Before ReLU: {hidden1}\n\n")
hidden1 = nn.ReLU()(hidden1)
print(f"After ReLU: {hidden1}")

# Sequential is an ordered container of modules
# Data is passed through the modules in the same order that they're defined
seq_modules = nn.Sequential(
    flatten,
    layer1,
    nn.ReLU(),
    nn.Linear(20, 10)
)
input_image = torch.rand(3, 28, 28)
logits = seq_modules(input_image)

# Logits are values from (-infty, +infty)
# Softmax scales these values to values between [0,1]
# Dim indicates the dimension along which the values must sum to 1
softmax = nn.Softmax(dim=1)
pred_probab = softmax(logits)


# Many layers inside a nn are parameterized (they have weights and biases optimized during training)
# Subclassing nn.Module makes all params accessible using model's parameters() or named_parameters() methods
print("\n\nModel Structure: ", model, "\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values: {param[:2]} \n")


