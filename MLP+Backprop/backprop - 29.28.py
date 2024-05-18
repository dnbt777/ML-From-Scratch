import numpy as np



@np.vectorize
def relu(x):
    return 0 if x < 0 else x

@np.vectorize
def relu_derivative(x):
    return 0 if x < 1 else 1

@np.vectorize
def sigmoid(X):
    return 1 / (1 + np.exp(-X))

@np.vectorize
def sigmoid_derviative(x):
    sig_x = sigmoid(x)
    return sig_x*(1-sig_x)



class MLP():
    def __init__(self, layer_counts):
        self.layers = [np.zeros((layer_count,)) for layer_count in layer_counts]
        self.weights = [np.random.rand(n*m).reshape(m, n) for n, m in zip(layer_counts, layer_counts[1:])]
        self.biases = [np.random.rand(layer_count) for layer_count in layer_counts[1:]]
        self.eta = 1e-3

        self.z_values = []
        self.weight_updates = []
        self.bias_updates = []


    def ff(self, x):
        for i in range(len(self.layers)):
            if i == 0:
                self.layers[i] = x
                self.z_values.append(x)
                continue
            W = i-1
            B = W
            Lm1 = i-1
            z = self.weights[W] @ self.layers[Lm1] + self.biases[B]
            self.z_values.append(z)

            if i < len(self.layers) - 1:
                self.layers[i] = relu(z)
            else:
                self.layers[i] = sigmoid(z) # short for last kayer

        yhat = self.layers[-1]
        return yhat
    # FF done in 8mins


    def backprop(self, y, yhat):
        # goal find dL_dz, then branch off to dL_dw dL_db for each
        for i in range(len(self.layers)):
            if i == len(self.layers) - 1:
                break
            L0 = len(layers) - 1 - i
            Lm1 = L0 - 1
            L1 = L0 + 1
            W0 = L0 - 1
            W1, Wm1 = W0+1, W0-1
            B0 = W0
            B1, Bm1 = W1, Wm1
            Z0 = L0

            if i == 0:
                # calculate from error
                dL_da0 = 2*(y - yhat) # (L0, 1)
                dL_dz0 = dL_da0 * (sigmoid_derviative(self.z_values[L0])) # (L0, 1)

            else:
                # dL_dz0 will already be defined
                dL_dz1 = dL_dz0 # (L1, 1) from last round
                dL_da0 = self.weights[W1].T @ dL_dz1 # (L1, L0) @ (L1, 1) => (L0, 0)
                if i != len(layers) - 1 - 1:
                    dL_dz0 = relu_derivative(self.z_values[L0])
                else:
                    dL_dz0 = self.layers[L0] #x

            
            dL_dw = np.outer(dL_dz0, self.layers[Lm1])
            dL_db = dL_dz0 * 1

            self.weight_updates.append(dL_dw)
            self.bias_updates.append(dL_db)
        

        # FLIP EM BOIS
        self.weight_updates = self.weight_updates[::-1]
        self.bias_updates = self.bias_updates[::-1]

    
    def update_and_zero_grad(self):
        #printshapes(self.weights)
        #printshapes(self.weight_updates)
        for i in range(len(self.weights)):
            self.weights[i] -= self.eta * self.weight_updates[i]
            self.biases[i] -= self.eta * self.bias_updates[i]
        self.weight_updates = []
        self.bias_updates = []



def printshapes(arr):
    print(" ".join(str(x.shape) for x in arr))


def MSE(y, yhat):
    return (np.sum(y) - np.sum(yhat))**2


# Begin tests
layers = [7, 2, 13, 4, 21]
x, y = np.random.rand(layers[0]), np.random.rand(layers[-1])
model = MLP(layers)

running_avg_losses = []
for i in range(10000):
    yhat = model.ff(x)
    loss = MSE(y, yhat)
    running_avg_losses.append(loss)
    model.backprop(y, yhat)
    model.update_and_zero_grad()
    if i%100 == 0 and i != 0:
        print("Loss: ", np.average(running_avg_losses))
        running_avg_losses = []
    


