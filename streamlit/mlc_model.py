import torch
import torch.nn as nn

# Multi-Class ANN Model (No softmax, ReLU in hidden layers)
class ANNModel(nn.Module):
    def __init__(self, input_size, output_size):
        super(ANNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 32)
        self.fc3 = nn.Linear(32, output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))  # Hidden Layer 1 + ReLU
        x = self.relu(self.fc2(x))  # Hidden Layer 2 + ReLU
        x = self.fc3(x)             # Output Layer (raw logits)
        return x
    
# model = ANNModel(input_size=10, output_size=3)  