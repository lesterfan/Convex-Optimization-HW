from main import *

def matrix_swap_test():
    test = np.array([1, 2, 4]).reshape(-1, 1)
    known = [1, 2]
    unknown = [0]
    print(matrix_swap(test, known + unknown))

    test = np.array([  [1, 4, 6], [4, 2, 5], [6, 5, 3]   ] )
    known = [0, 2]
    unknown = [1]
    test = matrix_swap(test, known + unknown)
    print(test)
    print(matrix_swap(test, known + unknown))

def missing_data_test_small():
    d = 3
    num_points = 8
    X = np.array([
        [1, 2, 3],
        [2, 2, 2],
        [5, 6, 7],
        [8, 9, 10],
        [-1, -1, -1],
        [-1, -1, -2],
        [-1, -1, -3],
        [-1, -2, -3]
    ], dtype = 'float')
    xs_input = [x.reshape(-1, 1) for x in X]
    ys_input = [1, 1, 1, 1, -1, -1, -1, -1]
    ms = [ set( [random.choice(range(d))] ) for _ in range(num_points) ]
    xs_result, ys_derived, covariances = estimate_missing_data(xs_input, ys_input, ms, mu_rtol=0.25, mu_atol=0.25, cov_rtol=0.5, cov_atol=0.5)
    print("Done!")

def missing_data_test_large():
    a = -100
    b = 100
    num_points = 500
    d = 7
    # Simulate x ~ Uniform(a, b)
    xs_input = [(b - a) * np.random.random_sample( (d, 1) ) + a for _ in range(num_points)]
    ys_input = [random.choice([-1, 1]) for _ in range(num_points)]
    ms = [ set( [random.choice(range(d))] ) for _ in range(num_points) ]
    xs_result, ys_derived, covariances = estimate_missing_data(xs_input, ys_input, ms, mu_rtol=1, mu_atol=1, cov_rtol=10, cov_atol=2)
    print("Done!")