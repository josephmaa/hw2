#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 18:54:16 2021

@author: Wenrui Li
"""

import numpy as np                 # Numpy is a library support computation of large, multi-dimensional arrays and matrices.
from PIL import Image              # Python Imaging Library (abbreviated as PIL) is a free and open-source additional library for the Python programming language that adds support for opening, manipulating, and saving many different image file formats.
import matplotlib.pyplot as plt    # Matplotlib is a plotting library for the Python programming language.

def betterSpecAnal(im: np.ndarray, block_size: int = 64, num_windows: int = 5) -> None:
    """
    Calculate the power spectrum density for the image.
    """
    height, width = im.shape
    # print('h', 'w', height, width)
    # Calculate the central window.
    block_width = block_size//2
    height_center, width_center = height//2-block_width, width//2-block_width
    block_ids = range(-num_windows//2, -num_windows//2 + num_windows)

    # Calculate the offsets for each of the blocks.
    block_offsets = [[height_idx * block_size, width_idx * block_size] for height_idx in block_ids for width_idx in block_ids]

    # Calculate the outer product for the hamming window.
    W = np.outer(np.hamming(block_size), np.hamming(block_size))
    # print(W)

    Z = np.zeros((block_size, block_size))
    
    for dy, dx in block_offsets:
        y, x = height_center + dy, width_center + dx
        # print(y, x)
        z = W * im[y: y + block_size, x: x + block_size]
        # Calculate the power spectrum for the window.
        Z += np.square(abs(np.fft.fft2(z)) / block_size)
    
    # Use fftshift to move the zero frequencies to the center of the plot.
    Z = np.fft.fftshift(Z)

    # Compute the logarithm of the Power Spectrum.
    Zabs = np.log(z)

    # Plot the result using a 3-D mesh plot and label the x and y axises properly. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, num = block_size)
    X, Y = np.meshgrid(a, b)

    surf = ax.plot_surface(X, Y, Zabs, cmap=plt.cm.coolwarm)

    ax.set_xlabel('$\mu$ axis')
    ax.set_ylabel('$\\nu$ axis')
    ax.set_zlabel('Z Label')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()

# Read in a gray scale TIFF image.
im = Image.open('img04g.tif')
# print('Read img04.tif.')
# print('Image size: ', im.size)

# Display image object by PIL.
# im.show(title='image')

# Import Image Data into Numpy array.
# The matrix x contains a 2-D array of 8-bit gray scale values. 
x = np.array(im)
print('Data type: ', x.dtype)

# Display numpy array by matplotlib.
# plt.imshow(x, cmap=plt.cm.gray)
# plt.title('Image')

# # Set colorbar location. [left, bottom, width, height].
# cax =plt.axes([0.9, 0.15, 0.04, 0.7]) 
# plt.colorbar(cax=cax)
# plt.show()

# x = np.double(x)/255.0
betterSpecAnal(x)



