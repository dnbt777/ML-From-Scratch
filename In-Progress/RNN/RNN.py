import numpy as np




class RNN():
    def __init__(self, rnn_setup):
        self.layer_sets = []
        previous_layer_set = None
        for layer_sizes in rnn_setup:
            layer_set = LayerSet(layer_sizes, previous_layer_set=previous_layer_set)
            self.layer_sets.append(layer_set)
            previous_layer_set = layer_set
    
    def __str__(self):
        return "".join([str(ls) for ls in self.layer_sets])


class LayerSet():
    def __init__(self, layer_sizes, previous_layer_set=None):
        self.layer_sizes = layer_sizes
        self.layers = [np.zeros((layer_size,)) for layer_size in layer_sizes]
        self.weights = [np.random.rand(n*m).reshape(n, m) for n, m in zip(layer_sizes, layer_sizes[1:])]
        self.biases = [np.random.rand(layer_size) for layer_size in layer_sizes]
        self.transition_weights = []

        if previous_layer_set:
            # make transition weights for all possible transitions
            previous_layer_sizes = previous_layer_set.layer_sizes
            for n0, nm1 in zip(previous_layer_sizes, layer_sizes):
                transition_weights_nm1_to_n0 = np.random.rand(nm1*n0).reshape(n0, nm1) # from n0 to nm1
                self.transition_weights.append(transition_weights_nm1_to_n0)


    def __str__(self):
        pad = lambda s, pad1, pad2: " "*pad1 + s + " "*pad2

        repr_layers = [f"[{layer.shape[0]}]" for layer in self.layers]
        repr_weights = [f"--{weight.shape}->" for weight in self.weights]
        repr_transition_weights = [f"{weight.shape}" for weight in self.transition_weights]

        # get layer set string

        layer_string = pad(repr_layers[0], 4, 1)
        for lstring, wstring in zip(repr_layers[1:], repr_weights):
            layer_string += pad(wstring, 2, 2) + pad(lstring, 2, 2)

        
        # get transition set string
        transition_strings = repr_transition_weights
        rest_transition_weight_padding = 8
        start_transition_weight_padding = 0
        for i in range(len(repr_transition_weights)):
            if i == 0:
                transition_weight_padding = start_transition_weight_padding
            else:
                transition_weight_padding = rest_transition_weight_padding
            arrow_padding = 5
            transition_weight_row_padding = int((arrow_padding*2 - len(repr_transition_weights[i]))/2)
            strings = [
                pad("^", transition_weight_padding + arrow_padding, arrow_padding),
                pad("|", transition_weight_padding + arrow_padding, arrow_padding),
                pad(repr_transition_weights[i], transition_weight_padding + transition_weight_row_padding, transition_weight_row_padding+1),
                pad("|", transition_weight_padding + arrow_padding, arrow_padding),
                pad("|", transition_weight_padding + arrow_padding, arrow_padding),
            ]
            transition_strings[i] = strings
        transition_string = "\n"
        if not len(transition_strings) == 0:
            for j in range(len(transition_strings[0])):
                for i in range(len(transition_strings)):
                    transition_string += transition_strings[i][j]
                transition_string += "\n"
        
        return transition_string + layer_string

                


                






rnn_setup = [
            [3,    5, 7, 9,    0],
            [0,    4, 6, 2,    7],
            [0,    1, 2, 3,    7],
            [0,    4, 6, 2,    7],
            [0,    4, 6, 2,    7],
]

model = RNN(rnn_setup)

print(model)