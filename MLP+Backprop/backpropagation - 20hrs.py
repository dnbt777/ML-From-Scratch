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

        self.weight_updates = [np.zeros_like(weight) for weight in self.weights]
        self.bias_updates = [np.zeros_like(bias) for bias in self.biases]

        self.eta = 1e-3
    

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
            self.layers[i+1] = np.array(neurons) #bruh
        
        if train:
            return neurons, activations, z_values
        return neurons
    

    def __repr__(self):
        return f"Input size: {self.input_size}\n" +\
            f"Weights: {[weight_matrix.shape for weight_matrix in self.weights]}\n"    +\
            f"Biases: {[bias_vector.shape for bias_vector in self.biases]}\n"


    def update_wandb(self):
        for W in range(len(self.weights)):
            self.weights[W] -= self.eta*self.weight_updates[W]

        for B in range(len(self.biases)):
            self.biases[B] -= self.eta*self.bias_updates[B]


    def zero_grad(self):
        self.weight_updates = [np.zeros_like(weight) for weight in self.weights]
        self.bias_updates = [np.zeros_like(bias) for bias in self.biases]


    def train_on_single_ff(self, x, y_target, eta=1e-4, debug=False):
        if not debug:
            print = lambda *x: None
            printshapes = print


        y_hat, activations, z_values = self.forward(x, train=True)
        activations = self.layers
        loss = self.loss_function(y_hat, y_target)

        weight_updates = []
        bias_updates = []
        # Now implement for the other layers
        for i in range(len(layers)): # going backwards
            L = len(layers) - i - 1                 # [L0 L1 L2 L3 L4]
            W = L - 1 # index for current weights   # [ w0 w1 w2 w3  ]
            B = L - 1 # index for current biases    # [   B0 B1 B2 B3]
            Z = L - 1 # no first z                  # [   z0 z1 z2 z3]
            ACT = L # one activation for each layer # [x  a1 a2 a3 a4]  # contains x here
            PREV_ACT = ACT - 1 # 

            if i == 0:
                # get dL/doutput_activation
                dL_doutputactivation = 2*(y_hat - y_target) # (output_layer, 1)

                # get d_output_activation/d_output_z
                # sigmoid for last layer relu for the rest
                output_z = z_values[Z]
                print(sum(output_z), [len(z) for z in z_values])
                doutputactivation_doutputz = sigmoid(output_z)*(1-sigmoid(output_z)) # (output_layer, 1)

                # get d_output_z/d_weights
                # also get d_output_z/d_biases (1)
                # this is just the last layer's activation
                previous_layer_activation = activations[PREV_ACT]
                doutputz_dweights = previous_layer_activation #(m, 1)
                doutputz_dbiases = 1


                # Calcualte the current delta
                # (L2, 1)
                printshapes(dL_doutputactivation, doutputactivation_doutputz)
                dL_dz = dL_doutputactivation*doutputactivation_doutputz

                # Calculate weight and bias updates from the delta dL_dz
                # (m, n) <= (n, 1) (n, 1) (1, m)
                # dl/dw = elementwise_mult(dl/doa * doa/doz) (x) doutputz_dweights.T
                dL_dweights = np.outer(dL_dz, doutputz_dweights)
                dL_dbiases = dL_dz*doutputz_dbiases

                print(dL_dweights.shape)
                print(dL_dbiases.shape)

                
            elif PREV_ACT >= 0:
                # L2 is the last layer in FF, L1 is the first layer of FF, L0 is the first.
                W2 = W+1
                ACT1 = L
                ACT0 = ACT1 - 1

                # Size (L2, 1)
                dL_dz2 = dL_dz
                
                
                dz2_da1 = self.weights[W2] # size (L1, L2) (top/input, side)
                printshapes(dz2_da1, dL_dz2)
                dL_da1 = np.dot(dz2_da1.T, dL_dz2)  # size (L1, L2) @ (L2, 1) => (L1, 1)

                
                z1 = z_values[Z] # size (L1, 1)
                da1_dz1 = relu_derivative(z1) # # size (L1, 1)
                printshapes(dL_da1, da1_dz1)
                dL_dz1 = dL_da1 * da1_dz1 # # size (L1, 1) * (L1, 1) => (L1, 1)
                dL_dz = dL_dz1 # Update the delta - now (L1, 1)

                # get branches of the chain leading to W and B
                dL_dbiases = dL_dz1 # size (L1, 1)

                dz1_dw1 = self.layers[ACT0] # size (L0, 1)
                print(ACT0, dz1_dw1.shape, [j.shape for j in activations])
                dL_dweights = np.outer(dL_dz1, dz1_dw1) # size (L1, 1) (x) (L0, 1) => (L0, L1)
            else:
                break

            print("Weights", dL_dweights.shape)
            print("Biases", dL_dbiases.shape)

            # store the gradients
            self.weight_updates[W] += dL_dweights
            self.bias_updates[B] += dL_dbiases
            weight_updates.append(dL_dweights) # muda
            bias_updates.append(dL_dbiases) # muda
            print(f"Weights {W} updated {dL_dweights.shape}")

        return loss




def printshapes(a, b):
    print("Shapes: ", a.shape, b.shape)


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
ffs = 1000
for i in range(ffs):
    loss = model.train_on_single_ff(x, y_target) # overfit to start (thanks andrej)
    model.update_wandb()
    model.zero_grad()

    if i%100 == 0:
        print(loss)



# tactic
# write out my current understanding of {paper}, then ask the model to correct it.