# import mnist data
# no it doesnt use numpy only it uses numpy + python-mnist
from mnist import MNIST
mndata = MNIST('samples')
train_images, train_labels = mndata.load_training()
test_images, test_labels = mndata.load_testing()


# from scratch now starts
import numpy as np

# create convNN
# how does this work again? standard NN but trains kernel 'weights' and biases?
# but what is the derivative of the kernel 'weights'??? fug
# z1 = convolve(kernel, a0)
# dz1_dz0 = ???
# figure out later, do ff for now

# idea: convNN BUT have a weight matrix on top of the image being convolved, elementwise multiply, and then do convolutin?

# for a model - apply random changes to the weights/biases (from a norm distribution), and make the backprop find the GENERAL location of local minima of loss

layer_sizes = [
    (781, 1),
    (50, 55),
    (70, 77),
    (90, 99),
    (20, 22),
    (10, 1),
]

class CNN():
    def __init__(self, layer_sizes):
        self.layers = [np.random.rand(n*m).reshape(n, m) for n, m in layer_sizes]
        self.weights = []