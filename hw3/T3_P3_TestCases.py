import numpy as np
import torch

'''
Instructions for using this Autograder:
0. In this problem, this autograder is used as a sanity check for your implementation of the forward pass of the neural network.
1. Make sure it is in the same immediate directory/folder as your implementation file, which *must* be called T3_P3.py
2. Run this only after you have implemented the forward pass (net) of your neural network.
'''

def test_forward_pass(net):
    sample_image = torch.zeros(1, 1, 28, 28)
    output = net(sample_image)
    assert output.shape == (1, 10), "Output shape is incorrect. Expected (1, 10), got {}".format(output.shape)

    print("Forward pass test passed!")