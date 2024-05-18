import numpy as np


class SimpleRNN():
    def __init__(self, input_size, hidden_size, output_size):
        self.input_layer = np.zeros(input_size)
        self.hidden_layer = np.random.rand(hidden_size)
        self.output_layer = np.zeros(output_size)

        self.layers = [self.input_layer, self.hidden_layer, self.output_layer]
        self.layer_sizes = [len(layer) for layer in self.layers]

        self.hidden_layers = [self.hidden_layer] # setup for future when more hidden layers
        self.hidden_layer_sizes = [len(layer) for layer in self.hidden_layers]


        self.layer_weights = [np.random.rand(n, m) for n, m in zip(self.layer_sizes, self.layer_sizes[1:])]
        self.recurrent_weights = [np.random.rand(h, h) for h in self.hidden_layer_sizes]

        self.layer_biases = [np.random.rand(l) for l in self.layer_sizes[1:]]
    



    def step(x):
        # get hidden layer state