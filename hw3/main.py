import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from cvxopt import matrix, solvers

def imshow_multiple(imgs, titles, m, n):
    assert(len(imgs) == m*n)
    fig = plt.figure()
    for i in range(m*n):
        fig.add_subplot(m, n, i + 1)
        plt.title(titles[i])
        plt.imshow(imgs[i], cmap='gray')
    plt.show()
    return fig

def problem1():
    img = mpimg.imread('cguitar.tif')
    # Use cvxopt to obtain parameters 
    # (See pdf for formulation of the quadratic program)
    t = 255
    def v(r, c):
        return np.array([r, c, 1], dtype = 'float')[:, np.newaxis]
    m = 250
    n = 50
    div_factor = 1e-11
    P = 2 * t**2 * sum([ div_factor * v(r, c) @ v(r, c).T for r in range(m) for c in range(n) ]) 
    q = -2 * t * sum([ div_factor * img[r, c] * v(r, c) for r in range(m) for c in range(n) ]) 
    G = np.array([
        [1, 1, 1],
        [-1, -1, -1]
    ], dtype = 'float')
    h = np.array([1, 0], dtype='float')[:, np.newaxis]
    sol = solvers.qp(*(matrix(M) for M in (P, q, G, h)))
    x = sol['x']
    print(x)
    # Actual values: x = [-1.00e-3, -2.00e-3, 10e-1]
    # Obtain true image
    def R(r, c):
        return x[0]*r + x[1]*c + x[2]
    true_img = np.copy(img)
    for r in range(img.shape[0]):
        for c in range(img.shape[1]):
            true_img[r, c] = img[r, c] / R(r, c)
    # Show the two images side by side
    imshow_multiple(
        [img, true_img],
        ["Original", "Corrected"],
        1, 2
    )
    
def problem2():
    img = mpimg.imread("curvedriver_wikipedia.jpg")
    def ginput_data():
        plt.imshow(img)
        x = plt.ginput(n = 0, timeout = 0)
        plt.show()
        for e in x:
            print(e)
    # ginput_data()
    points = []
    with open("river_data.txt", "r") as f:
        for line in f:
            str_list = line.strip().split(" ")
            num_list = [
                float(str_list[0][1:-1]),
                float(str_list[1][:-1])
            ]
            points.append(num_list)

if __name__ == "__main__":
    # problem1()
    problem2()