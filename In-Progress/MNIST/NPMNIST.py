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


# convolves in the time domain (inefficient on^2 type beat)
def convolve_time(data, kernel, kernel_bias, stride=1, dilation=0, padding=0):
    # define kernel 'pointer' start/end locations (pointer is top left square)
    kernel_start_y = 0
    kernel_end_y = (len(data) - 1) - (len(kernel) - 1)
    kernel_start_x = 0
    kernel_end_x = (len(data[0]) - 1) - (len(kernel[0]) - 1)
    convolution = np.zeros((kernel_end_y, kernel_end_x))
    kernel_cache = np.zeros_like(kernel)

    # yes I know how this looks:
    for j in range(kernel_end_y):
        for i in range(kernel_end_x):
            kernel_pointer = (j, i)
            # uh oh
            for kj in range(len(kernel)):
                for ki in range(len(kernel)):
                    kernel_cache[j][i] = kernel[kj][ki] * data[j][j] + kernel_bias[kj][ki]
            
            convolution[j][i] = np.sum(kernel_cache) # sums all dims into one

    return convolution




# convolves in the time domain (inefficient on^2 type beat)
def convolve_fft(data, kernel):
    padded_kernel = np.zeros_like(data)
    padded_kernel[:kernel.shape[0]][:kernel.shape[1]] = kernel

    data_fft = np.fft.fft2(data)
    kernel_fft = np.fft.fft2(padded_kernel)

    convolution_fft = data_fft * kernel_fft
    convolution = np.real(np.fft.ifft2(convolution_fft))
    # trim edges? nah not rn tbh
    # hmm this is flawed, cant do biases afaik.....
    return convolution



class InputLayer():
    def __init__(self, size):
        self.size = size


# single kernel, single input layer for now
# in future: list of kernels, list of input layers, output = layers x kernels
class ConvLayer():
    def __init__(self, size, kernel_size, layer_count=1, kernel_count=2):
        self.layer_count = layer_count
        self.kernel_count = kernel_count
        self.layers = [np.zeros(size) for _ in range(layer_count)]
        self.kernels = [np.random.rand(*kernel_size) for _ in range(kernel_count)]
        self.kernel_biases = [np.random.rand(*kernel_size) for _ in range(kernel_count)]
        self.convolution_func = convolve_time
    
    # this is just one layer, not a network! 
    def ff(self):
        output_layers = []
        for layer in self.layers:
            for kernel in self.kernels:
                output = convolve_time(layer, self.kernel, self.kernel_biases)
                output_layers.append(output)
        return output

    def backprop():
        # take dL_dz1 from future layer
        # use to calculate dL_dz0
        # from there calculate wandb
        # store the dL_dz0 somewhere for the next delta
        pass

# isnt this just convlayer w large kernel? oh wait but the kernel is just 1111111 or something so it averages out? pixel = avg(kernelxpixels)
# oh 'max' pool. pool by max. lol.
class MaxPoolLayer():
    def __init__(self, input_size, output_layer_size):
        self.input_size = input_size
        self.output_layer_size = output_layer_size # only need x and y bc layer num is predetermined
        self.output_size = (input_size[0], output_layer_size[0], output_layer_size[1]) # no of layers
        self.layers = np.zeros(input_size)
        self.kernel_size = (
            self.input_size[0] - self.output_size[0],
            self.input_size[1] - self.output_size[1],
        )
        self.output = np.zeros(self.output_size)
    

    def maxpool(self):
        for L in range(self.layers):
            for j in range(self.output_size[0]):
                for i in range(self.output_size[1]):
                    maxpool = max(self.layers[L][:j+self.kernel_size[0]][:i+self.kernel_size[1]])
                    self.output[j][i] = maxpool
    




    


class FF():
    def __init__(self, size):
        pass
        

L1 = ConvLayer((28, 28), (2, 2), layer_count=1, kernel_count=2)
L2 = MaxPoolLayer(
    (L1.layer_count*L1.kernel_count, L1.size[0], L1.size[1])
)


layers = [
    L1,
    MaxPoolLayer((28-2, 28-2, ), (70, 77)), # layerm1 - kernelm1, layerm1 - kernelm1, kernelsm1
    ConvLayer((90, 99)),
    MaxPoolLayer((20, 22)),
    FF((10, 1)),
]

class CNN():
    def __init__(self, layer_sizes):
        self.layers = [np.random.rand(n*m).reshape(n, m) for n, m in layer_sizes]
        self.weights = []