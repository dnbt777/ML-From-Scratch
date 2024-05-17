from NPMNIST import *

import numpy as np

class MaxPoolLayer:
    def __init__(self, input_size, output_layer_size):
        self.input_size = input_size
        self.output_layer_size = output_layer_size  # only need x and y bc layer num is predetermined
        self.output_size = (input_size[0], output_layer_size[0], output_layer_size[1])  # no of layers
        self.layers = np.random.rand(*input_size)  # Initialize with random values for testing
        self.kernel_size = (
            self.input_size[1] - self.output_size[1],
            self.input_size[2] - self.output_size[2],
        )
        self.output = np.zeros(self.output_size)

    def maxpool(self):
        for L in range(self.input_size[0]):
            for j in range(self.output_size[1]):
                for i in range(self.output_size[2]):
                    maxpool = np.max(self.layers[L, j:j+self.kernel_size[0]+1, i:i+self.kernel_size[1]+1])
                    self.output[L, j, i] = maxpool

# Test the MaxPoolLayer class
input_size = (1, 4, 4)  # 1 layer, 4x4 input
output_layer_size = (2, 2)  # 2x2 output
max_pool_layer = MaxPoolLayer(input_size, output_layer_size)
max_pool_layer.maxpool()

print("Input Layers:")
print(max_pool_layer.layers)
print("\nMax Pooled Output:")
print(max_pool_layer.output)