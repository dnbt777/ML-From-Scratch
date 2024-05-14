import numpy as np


class MLP():
    def __init__(self, layers):
        self.input_size = layers[0]
        self.weights = [
            np.random.randn(*(dim_out, dim_in)) for dim_in, dim_out in zip(layers[:-1], layers[1:])
        ]
        self.biases = [
            np.random.randn(*(n,)) for n in layers[1:]
        ]
        self.layers = [np.zeros((layer_size,)) for layer_size in layers]
        self.activation_function = np.vectorize(lambda x: max(0, x)) # relu
        self.output_activation_function = sigmoid
        self.loss_function = messi # implement MSE here
    

    def forward(self, x, train=False):
        # assume x is input size
        neurons = x
        activations = [neurons] # used in backpropagation
        self.layers[0] = np.array(neurons)
        z_values = []
        for i, layer in enumerate(self.layers[1:]):
            weights = self.weights[i]
            biases = self.biases[i]
            z = np.matmul(weights, neurons)
            z = z + biases
            z_values.append(z) # used for backpropogation
            if i == len(layers) - 1:
                neurons = self.output_activation_function(z)
            else:
                neurons = self.activation_function(z)
            activations.append(neurons) # used in backpropogation to calculate d_Z/d_W
        # calculate output activations
            self.layers[i] = np.array(neurons)
        
        if train:
            return neurons, activations, z_values
        return neurons
    

    def __repr__(self):
        return f"Input size: {self.input_size}\n" +\
            f"Weights: {[weight_matrix.shape for weight_matrix in self.weights]}\n"    +\
            f"Biases: {[bias_vector.shape for bias_vector in self.biases]}\n"




    def train_on_single_batch(self, x, y_target, eta=1e-4):
        y_hat, activations, z_values = self.forward(x, train=True)
        loss = self.loss_function(y_hat, y_target)

        weight_updates = []
        bias_updates = []

        # Now implement for the other layers
        for i in range(len(layers), 0, -1): # going backwards
            if i == len(layers):
                # get dL/doutput_activation
                dL_doutputactivation = 2*(y_hat - y_target) # (output_layer, 1)

                # get d_output_activation/d_output_z
                # sigmoid for last layer relu for the rest
                output_z = z_values[-1]
                doutputactivation_doutputz = sigmoid(output_z)*(1-sigmoid(output_z)) # (output_layer, 1)

                # get d_output_z/d_weights
                # also get d_output_z/d_biases (1)
                # this is just the last layer's activation
                previous_layer_activation = activations[-2]
                doutputz_dweights = previous_layer_activation #(m, 1)
                doutputz_dbiases = 1

                # (m, n) <= (n, 1) (n, 1) (1, m)
                # dl/dw = elementwise_mult(dl/doa * doa/doz) (x) doutputz_dweights.T
                dL_dweights = np.outer(dL_doutputactivation*doutputactivation_doutputz, doutputz_dweights)
                dL_dbiases = dL_doutputactivation*doutputactivation_doutputz*doutputz_dbiases

                print(dL_dweights.shape)
                print(dL_dbiases.shape)

                current_delta = dL_doutputactivation*doutputactivation_doutputz
            else:
                # update delta to next chain
                # get branches of the chain leading to W and B
                # add w and b to updates lists

            
            # store the gradients
            weight_updates.append(dL_dweights)
            bias_updates.append(dL_dbiases)



        return





        # Old attempt/notes
        # backpropogation starts here
        # start with last weights and biases
        # goal: find d_loss / d_weights, and d_loss / d_biases
        # gonna start by listing the partial derivatives of each component
        # loss = target - output
            # output = relu(weights_n(activation_n-1))    # relu or sigmoid, whatever
        # d_loss / d_weights = d_loss / activation_n * output_n / d_weights
        #
        # d_L/d_W = d_L/d_output * d_output/d_z * d_z/d_W
        # d_loss / d_output = 2*(output_y - target_y) # if MSE
        # d_output/d_z = ??? (depends on the activation function) if relu: {1 if z>0, 0 otherwise}
        #     d_output/d_z sigmoid: sigmoid(z)*(1-sigmoid(z))
        #     d_output/d_z tanh: 1 - tanh^2(z)
        #     d_output/d_z linear (no activation): 1
        # d_z/d_W = layer_n-1

        # Something like this I think
        # d_L___d_output = 2*(output_y - target_y)
        # d_z___d_W = x # its just the last layer bro
        # activation = "sigmoid"
        # if activation=="relu":
        #     pass
        # if activation=="sigmoid":
        #     d_output_over_d_z


       

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def messi(y_hat, y_target):
    assert len(y_hat) == len(y_target), "MSE: Y lengths do not match"
    n = len(y_hat)
    mse = np.vectorize(lambda yhat, ytarget: (ytarget-yhat)**2)(y_hat, y_target)
    mse = sum(mse)/n
    return mse

@np.vectorize
def relu_derivative(x):
    if x > 0:
        return 1
    return 0

@np.vectorize
def sigmoid_derivative(x):
    return sigmoid(x)*(1-sigmoid(x))


if False:
    mse_test = messi([1, 2, 4], [1, 1, 1])
    print(mse_test)

if False:
    layers = [7, 2, 13, 4, 21]
    model = MLP(layers)
    x = np.ones((layers[0],))
    print(model)
    y = model.forward(x)
    print("Example forward pass output:", y)


print("Begin training run")
# goal should be to have the model output all ones (for now!)
layers = [7, 2, 13, 4, 21]
model = MLP(layers)
x = np.ones((layers[0],))
y_target = np.ones(layers[-1],)
print(model)
# start training
model.train_on_single_batch(x, y_target)



# tactic
# write out my current understanding of {paper}, then ask the model to correct it.