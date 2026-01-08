import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sys import argv


class Net(nn.Module):
    def __init__(self, input_dim, hidden_dim, number_tokens, norm=1.0, beta=1.0):
        super(Net, self).__init__()

        self.beta = beta
        self.D = input_dim
        self.L = number_tokens 
        self.R = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=False)  # First layer
        self.fc1.weight.data.normal_(0, norm)  # Initialize weights

    def forward(self, x):
        x = self.fc1(x) / np.sqrt(self.D)  # Normalize by sqrt(D), x becomes (batch_size, L, R)

        attention_matrix = torch.einsum('nap,nbp->nab', x, x) / np.sqrt(self.R)
        
        trace_part = torch.norm(self.fc1.weight)**2/np.sqrt(self.R * self.D**2) # Trace part
        x = attention_matrix - trace_part * torch.eye(self.L)  # Subtract trace part
        x = nn.Softmax(dim=-1)(self.beta * x)  # Apply softmax to the attention matrix
        return x
    

def GD_run_averaged(D, L, alpha, rho, T, averages, learning_rate=0.02, norm_init=1.0):
    N = int(alpha * D**2)
    R = int (rho * D)

    x_train = torch.normal(0,1, (N, L, D))

    with torch.no_grad():
        teacher = Net(D, R, L, 1.0)
        y_train = teacher(x_train)

    W_teacher = teacher.fc1.weight.data.numpy()
    S_teacher = W_teacher.T @ W_teacher / np.sqrt(R * D)

    S_list = np.zeros((averages, D, D))
    for a in range(averages):
        student = Net(D, R, L, norm_init)
        optimizer = torch.optim.Adam(student.parameters(), lr=learning_rate)

        for _ in range(T):
            optimizer.zero_grad()
            y_pred = student(x_train)
            
            
            loss = torch.sum((y_pred - y_train)**2)/D
            
            loss.backward()
            optimizer.step()
        
        W_student = student.fc1.weight.data.numpy()
        S_student = W_student.T @ W_student / np.sqrt(R * D)
        S_list[a] = S_student

        MSE_averaged = ((S_list[:a+1].mean(axis=0) - S_teacher)**2).sum() / D
        print(MSE_averaged)


    return MSE_averaged


D = 200

alpha_list = np.logspace(np.log10(0.0012), np.log10(0.3), 32)

alpha = alpha_list[int(argv[1])]

L = 2

rho = 0.5
learning_rate = 0.1
norm_init = 1.0
samples = 16
averages = 1

T = 1000

MSE_list = np.zeros((samples))
for i in range(samples):
    MSE_averaged = GD_run_averaged(D, L, alpha, rho, T, averages, learning_rate=learning_rate, norm_init=norm_init)
    MSE_list[i] = MSE_averaged

# Save MSE_list how you prefer!