import numpy as np # IT BEGINS
import random

# helper functions
@np.vectorize
def relu(x):
    if x < 0:
        return 0
    return x

@np.vectorize
def relu_derivative(x):
    if x < 0:
        return 0
    return 1

@np.vectorize
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


@np.vectorize
def sigmoid_derivative(x):
    return sigmoid(x)*(1 - sigmoid(x))

# f it not debugging
relu = sigmoid
relu_derivative = sigmoid_derivative


def printsizes(arr):
    print(" ".join([str(x.shape) for x in arr]))


class MLP():
    def __init__(self, layer_sizes):
        self.layers = [np.random.rand(layer_size) for layer_size in layer_sizes]
        self.weights = [np.random.rand(n*m).reshape(((m, n))) for n, m in zip(layer_sizes, layer_sizes[1:])]
        self.biases = [np.random.rand(layer_size) for layer_size in layer_sizes[1:]]

        self.weight_updates = []
        self.bias_updates = []
        self.activation_func = relu
        self.output_activation_func = sigmoid

        self.eta = 1e-3

        self.z_values = []




    def ff(self, x):
        for i, _ in enumerate(self.layers):
            L = i
            W = i-1 # no w for first layer
            B = i-1 # no b for first layer
            if i == 0:
                assert self.layers[L].shape == x.shape, "layer does not match input"
                self.layers[L] = x
                self.z_values.append(x)
                continue
            else:
                zw = self.weights[W] @ self.layers[L-1]
                zb = self.biases[B]
                z = zw + zb
                if L != len(self.layers) - 1:
                    self.layers[L] = self.activation_func(z)
                else:
                    self.layers[L] = self.output_activation_func(z)
                self.z_values.append(z)
        yhat = self.layers[-1] # get output layer
        return yhat
        

    def backprop(self, y, yhat):
        for i in range(len(self.layers)):
            L0 = (len(self.layers) - 1) - i
            L1, Lminus1 = L0+1, L0-1
            W0 = L0 - 1
            B0 = L0 - 1
            W1, Wminus1 = W0+1, W0-1
            B1, Bminus1 = B0+1, B0-1

            Z0 = L0

            # get to the next z0 in the chain
            if i == 0:
                dL_da0 = 2*(y - yhat) # TODO check if correct
                da0_dz0 = sigmoid_derivative(self.z_values[Z0])
                dL_dz0 = dL_da0 * da0_dz0
                delta = dL_dz0
            elif i == len((self.layers))-1:
                break
            else:
                dz1_da0 = self.weights[W1] # (L0, L1)
                dL_da0 = dz1_da0.T @ delta # (L0, L1) @ (L1, 1) => (L0, 1)

                da0_dz0 = relu_derivative(self.z_values[Z0]) # (L0, 1)
                dL_dz0 = dL_da0 * da0_dz0 # (L0, 1) - elementwise

                delta = dL_dz0

                # calculate delta as normal from the z0 'node'
            
            # branch from the z node in the chain, calculate dw and db
            dz0_dw0 = self.layers[Lminus1]# previous activation
            dz0_db0 = 1 # lol

            # calculate wandb updates
            dL_dw0 = np.outer(delta, dz0_dw0)
            dL_db0 = delta * dz0_db0

            #print(dL_da0, da0_dz0 dL_dz0 delta dz1_da0 dL_da0 dL_dz0 dz0_dw0 dz0_db0)
            self.weight_updates.append(dL_dw0)
            self.bias_updates.append(dL_db0)
        
        # flip em, bois
        self.weight_updates = self.weight_updates[::-1]
        self.bias_updates = self.bias_updates[::-1]


    def update_wandb(self):
        for W, _ in enumerate(self.weights):
            self.weights[W] += self.eta * self.weight_updates[W]
        for B, _ in enumerate(self.biases):
            self.biases[B] += self.eta * self.bias_updates[B]


    def zero_grad(self):
        self.weight_updates = []
        self.bias_updates = []


    def train(self, xs, ys, rounds=10000):
        train_data = list(zip(xs, ys))
        for i in range(rounds):
            x, y = random.choice(train_data)
            yhat = self.ff(x)
            loss = (np.sum(y) - np.sum(yhat))**2 # MESSI
            self.backprop(y, yhat)
            self.update_wandb()
            self.zero_grad()
            if i % 1000 == 0:
                print(loss)
        

layer_sizes = [20, 30, 50, 70, 110, 13]
layer_sizes = [10, 5, 7, 8, 10]
mlp = MLP(layer_sizes)
x, y = np.random.rand(layer_sizes[0]), np.random.rand(layer_sizes[-1])
xs, ys = [x], [y]
mlp.train(xs, ys)


x = """

# start w single head
class Transformer():
    def __init__(self, vocab,
                 model_size=512, # paper uses this
                 qkv_size=128, # paper I think?
                 model_heads=8,
                 ff_hidden_layers=[]
                 ):
        self.vocab = vocab # list of tokens
        self.model_size = 512 # channels for each embed

        self.embeddings = self.init_embeddings() # heads, 
        self.wq = np.random.rand((model_size, )) # 512 (input) x random size (output)(paper uses 128 I think?) 
        self.qk = 
        self.wv = [] # will need to rewatch vid 3b1b
        self.feedforward = MLP([model_size] + ff_hidden_layers + [model_size])

"""