import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from BetterSpecAnal import betterSpecAnal


def main():
    x = np.random.uniform(-0.5, 0.5, (512, 512))
    x_scaled = 255*(x+0.5)
    plt.imshow(x_scaled)
    plt.show()

    # Hmmm... this doesn't look exactly right
    height, width = x_scaled.shape
    y = np.zeros((height, width))
    for m in range(512):
        for n in range(512):
            y[m, n] = 3 * x_scaled[m, n]
            if m >= 1:
                y[m, n] += 0.99 * y[m-1, n]
            if n >= 1:
                y[m, n] += 0.99 * y[m, n-1]
            if m >= 1 and n >= 1:
                y[m, n] += -0.9801 * y[m-1, n-1]
    y + 127
    plt.imshow(y)
    # print(y)
    plt.show()

    N = 512
    Z = (1/N**2)*abs(np.fft.fft2(y))**2
    Z = np.fft.fftshift(Z)
    Zabs = np.log(Z)

    # Plot the result using a 3-D mesh plot and label the x and y axises properly. 
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    a = b = np.linspace(-np.pi, np.pi, 512)
    X, Y = np.meshgrid(a, b)

    surf = ax.plot_surface(X, Y, Zabs, cmap=plt.cm.coolwarm)

    ax.set_xlabel('$\mu$ axis')
    ax.set_ylabel('$\\nu$ axis')
    ax.set_zlabel('Z Label')

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


if __name__== "__main__":
    main()