import numpy as np


# Get data
datax, datay = 20, 30
data = np.random.rand(datax*datay).reshape((datax, datay))

# Make kernel
kernelx, kernely = 2, 2
kernel = np.random.rand(kernelx*kernely).reshape((kernelx, kernely))

# Pad kernel so we can multiply afterwards
padded_kernel = np.zeros_like(data)
padded_kernel[:kernelx, :kernely] = kernel # whoa!!


# compute ffts
data_fft = np.fft.fft2(data)
kernel_fft = np.fft.fft2(padded_kernel)

# convolve via multiplication
convolution_fft = data_fft*kernel_fft

# get the time domain convolution and clean for all reals
convolution = np.fft.ifft2(convolution_fft)
convolution = np.real(convolution)
print(convolution.size)