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
    x_list = []
    y_list = []
    with open("river_data.txt", "r") as f:
        for line in f:
            str_list = line.strip().split(" ")
            x_list.append(float(str_list[0][1:-1]))
            y_list.append(float(str_list[1][:-1]))
    def chunks(l, n):
        assert(len(l) % n == 0)
        for t in range(0, len(l), n):
            yield list(range(t, t + n))
    knot_interval = 6
    div_factor = 1
    def v(t):
        return np.array([1, t, t**2, t**3], dtype = 'float')[:, np.newaxis]
    def P(I):
        return 2 * sum([ v(t) @ v(t).T for t in I ])
    def q(S, I):
        return -2 * sum([ S[t] * v(t) for t in I ])
    def P_block(Ps, r, c):
        if r != c:
            return np.zeros(Ps[0].shape, dtype = 'float')
        return Ps[r]
    model_results = []
    for S in (x_list, y_list):
        Ps = []
        qs = []
        for I in chunks(S, knot_interval):
            Ps.append(P(I))
            qs.append(q(S, I))
        big_P = np.block([
            [P_block(Ps, r, c) for c in range(len(Ps))] for r in range(len(Ps))
        ])
        big_q = np.vstack(
            [q_I for q_I in qs]
        )
        sol = solvers.qp(*(matrix(M) for M in (big_P, big_q)))
        model_results.append(sol['x'])
    # Get spline points
    splines = []
    for i, S in enumerate([x_list, y_list]):
        curr_spline = []
        for j, I in enumerate(chunks(S, knot_interval)):
            coeffs = model_results[i][4*j : 4*(j + 1), 0]
            def model(t):
                return coeffs[0] + coeffs[1] * t + coeffs[2] * t**2 + coeffs[3] * t**3
            for t in I:
                curr_spline.append(model(t))
        splines.append(curr_spline)

    # Image and points.
    x_knots = [x_list[I[0]] for I in chunks(x_list, knot_interval)]
    y_knots = [y_list[I[0]] for I in chunks(y_list, knot_interval)]
    implot = plt.imshow(img)
    plt.scatter(x_list, y_list, c = 'r', s = 5)
    plt.scatter(splines[0], splines[1], c = 'g', s = 5)
    plt.scatter(x_knots, y_knots, c = 'b', s = 5)
    plt.show()

if __name__ == "__main__":
    # problem1()
    problem2()