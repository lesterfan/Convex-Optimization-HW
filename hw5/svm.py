import cvxpy as cp
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import random
from mpl_toolkits.mplot3d import axes3d
from multiprocessing import Process

def svm(x_arr, y_arr, gamma_vals):
    """
    Both x_arr and y_arr are lists of np.ndarrays with shape
    (m, 1).
    x_arr has length N, y_arr has length M
    """
    m = x_arr[0].shape[0]
    N = len(x_arr)
    M = len(y_arr)
    # Formulate optimization variables and parameters
    u = cp.Variable(N)
    v = cp.Variable(M)
    a = cp.Variable(m)
    b = cp.Variable()
    gamma = cp.Parameter(nonneg = True)
    # Formulate the optimization problem
    one_vec_u = np.ones(u._shape, dtype = np.float64)
    one_vec_v = np.ones(v._shape, dtype = np.float64)
    obj = cp.Minimize( cp.norm(a) + gamma * (one_vec_u.T * u + one_vec_v.T * v) )  
    constraints = []
    for i in range(N):
        constraints.append(
            a.T * x_arr[i] - b >= 1 - u[i]
        )
    for i in range(M):
        constraints.append(
            a.T * y_arr[i] - b <= -(1 - v[i])
        )
    constraints.append(u >= 0)
    constraints.append(v >= 0)
    prob = cp.Problem(obj, constraints)
    # Run the solver for each gamma in gamma_vals
    a_optimals, b_optimals = [], []
    for gamma_val in gamma_vals:
        gamma.value = gamma_val
        prob.solve()
        a_optimals.append(a.value)
        b_optimals.append(b.value)
        # print(f'For gamma = {gamma_val}, optimal vals are: a = {a.value}, b = {b.value}')
    return a_optimals, b_optimals

def imshow_multiple(imgs, titles, m, n):
    assert(len(imgs) == m*n)
    fig = plt.figure()
    for i in range(m*n):
        fig.add_subplot(m, n, i + 1)
        plt.title(titles[i])
        if len(imgs[i].shape) == 2:
            plt.imshow(imgs[i], cmap='gray')
        else:
            plt.imshow(imgs[i])
    plt.show()
    return fig

def classify_image_pixels(image_dir, x_pts, y_pts, 
                num_x = 500, num_y = 500, 
                color_target = [0, 0, 0], color_thresh = 10):
    """
    x_pts and y_pts are both lists of tuples of ints which specify coordinates in the image.
    x_pts are points that are supposed to be classified as 'yes', 
    y_pts are points that are supposed to be classified as 'no'
    num_x is the number of training points to use for x
    num_y is the number of training points to use for y
    if num_x > |x_pts| or num_y > |y_pts|, a random sampling will be used.
    """
    img = mpimg.imread(image_dir)
    def ginput_data():
        plt.imshow(img)
        x = plt.ginput(n = 0, timeout = 0)
        plt.show()
        for e in x:
            print(e)
    # Uncomment to get input data for x_pts and y_pts:
    # ginput_data()
    def get_feature_vector(r, c):
        curr_vector = np.zeros(shape = (5, 1), dtype = np.float64)
        curr_vector[0] = r
        curr_vector[1] = c
        curr_vector[2] = img[r][c][0]
        curr_vector[3] = img[r][c][1]
        curr_vector[4] = img[r][c][2]
        return curr_vector
    def get_feature_vectors(coords_list):
        result = []
        for r, c in coords_list:
            result.append(get_feature_vector(r, c))
        return result
    x_arr = get_feature_vectors(random.sample(x_pts, num_x))
    y_arr = get_feature_vectors(random.sample(y_pts, num_y))
    gamma_vals = [1]
    # Learned 
    a_optimals, b_optimals = svm(x_arr, y_arr, gamma_vals)
    learned_imgs = []
    learned_imgs_titles = []
    for gamma_val, a_val, b_val in zip(gamma_vals, a_optimals, b_optimals):
        def f(curr_pt):
            return a_val.T @ curr_pt - b_val
        def classify(r, c):
            curr_pt = get_feature_vector(r, c)
            return 0 if f(curr_pt) >= 0 else 255
        learned_img = np.array([
            [classify(r, c) for c in range(len(img[0]))]
            for r in range(len(img))
        ])
        curr_title = f'SVM on {image_dir} with gamma = {gamma_val}'
        learned_imgs.append(learned_img)
        learned_imgs_titles.append(curr_title)
        # Plot training data with learned classifier
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter([vec[2] for vec in x_arr], [vec[3] for vec in x_arr], [vec[4] for vec in x_arr], c = 'g', marker ='o', label = 'In boundary')
        ax.scatter([vec[2] for vec in y_arr], [vec[3] for vec in y_arr], [vec[4] for vec in y_arr], c = 'r', marker ='o', label = 'Not in boundary')
        rs = np.linspace(-10, 310, 1000)
        gs = np.linspace(-10, 310, 1000)
        R, G = np.meshgrid(rs, gs)
        B = (b_val - a_val[2] * R - a_val[3] * G) / a_val[4]
        ax.plot_surface(R, G, B)
        ax.set_xlabel('r')
        ax.set_ylabel('g')
        ax.set_zlabel('b')
        ax.legend()
        plt.title(curr_title)
    # Images found through just a color threshold
    def color_classify(r, c):
        if all([ 
            abs(color_target[i] - img[r][c][i]) <= color_thresh 
            for i in range(len(color_target))
            ]):
            return 0
        return 255
    naive_img = np.array([
        [color_classify(r, c) for c in range(len(img[0]))]
        for r in range(len(img))
    ])
    imgs = [img] + [naive_img] + learned_imgs
    titles = ["Original"] + [f'Color Seperation with Threshold {color_thresh}'] + learned_imgs_titles
    fig.show()
    imshow_multiple(imgs, titles, 1, 3)

def classify_apple():
    image_dir = "apple.jpg"
    x_pts = [(y, x) for x in range(153, 226) for y in range(36, 180)]
    y_pts = (
        [(y, x) for x in range(0, 117) for y in range(0, 150)] + 
        [(y, x) for x in range(208, 348) for y in range(1, 31)] + 
        [(y, x) for x in range(282, 367) for y in range(138, 202)] + 
        [(y, x) for x in range(1, 48) for y in range(146, 200)] 
    )
    classify_image_pixels(image_dir, x_pts, y_pts, color_target=[160, 31, 26], color_thresh=70)

def classify_hand():
    image_dir = "hand.jpg"
    x_pts = [(y, x) for x in range(69, 348) for y in range(89, 319)]
    y_pts = (
        [(y, x) for x in range(0, 61) for y in range(0, 90)] + 
        [(y, x) for x in range(412, 479) for y in range(2, 355)] 
    )
    classify_image_pixels(image_dir, x_pts, y_pts, color_target=[228, 184, 157], color_thresh=85)

if __name__ == "__main__":
    print("Starting!")
    p = Process(target=classify_apple)
    p.start()
    q = Process(target=classify_hand)
    q.start()