from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
import numpy as np
import random
import copy
import csv

def matrix_swap(np_arr, indices):
    """Swaps rows and cols of np_arr according to the order given by 
    indices.

    Args:
        np_arr (np.array) : the (2 dimensional) data 
        indices (list(int)) : the input indices
    Returns: 
        result (np.array) : the resulting data
    """
    assert( np_arr.shape[0] == np_arr.shape[1] or np_arr.shape[0] == 1 or np_arr.shape[1] == 1 )
    result = copy.deepcopy(np_arr)
    for r in range(result.shape[0]):
        for c in range(result.shape[1]):
            r_mapped = indices[r] if result.shape[0] > 1 else 0
            c_mapped = indices[c] if result.shape[1] > 1 else 0
            result[r][c] = np_arr[r_mapped][c_mapped]  
    return result

def estimate_missing_data(xs_input, ys_input, ms_input, 
                          mu_rtol = 1e-2, mu_atol = 1e-2, cov_atol = 1e-1, cov_rtol = 1e-1):
    """Estimates missing data given the available data {(x, y) | x in xs, y in ys}.
    ms contains vectors which indicate whether a value is known to be missing or not.
    Note that this function expects the data to be initialized regardless if it's known
    or not.

    Args:
        xs_input (list of np.arrays with shape (n, 1)) : the data
        ys_input (list of ints which take values from {-1, 1}) : the corresponding classes
        ms_input (list of sets which contain indices of missing data in x)

    Kwargs:
        mu_atol (float) : The absolute tolerance for determining convergence of mu
        mu_rtol (float) : The relative tolerance for determining convergence of mu
        cov_atol (float) : The absolute tolerance for determining convergence of the covariance
        cov_rtol (float) : The relative tolerance for determining convergence of the covariance

    Returns:
        Tuple of (xs_derived, covariances) where:
            xs_derived (a list of np.arrays with shape (n, 1)) : the derived data
            ys_derived (list of ints which take values from {-1, 1}) : the corresponding classes
            covariances (list of np.arrays of shape (n, n)) : covariance matrices, one for each
                                                              class (so two in total)
    """
    assert(len(xs_input) == len(ys_input) and len(xs_input) >= 1)
    xs_derived  = [None for _ in xs_input]
    ys_derived  = [None for _ in ys_input]
    covariances = []
    # Make a copy of the data which we will change so we don't corrupt the original variable
    xs_cpy = copy.deepcopy(xs_input)
    # Partition x's into x's of class -1 and x's of class 1
    data_partitioned = {-1 : ([], [], []), 1 : ([], [], [])}
    for i, (x, y, m) in enumerate(zip(xs_cpy, ys_input, ms_input)):
        data_partitioned[y][0].append(x)
        data_partitioned[y][1].append(m)
        data_partitioned[y][2].append(i)
    # Perform EM for each class
    for y, (xs, ms, indices) in data_partitioned.items():
        # X is the matrix where the rows are the feature vectors
        X = np.hstack(xs)
        mu = np.mean(xs, axis = 0)
        covariance = np.cov(X)
        mu_prev = np.zeros(mu.shape)
        covariance_prev = np.zeros(covariance.shape)
        test_ctr = 0
        while not (np.allclose(mu, mu_prev, rtol = mu_rtol, atol = mu_atol) and 
                   np.allclose(covariance, covariance_prev, rtol = cov_rtol, atol = cov_atol)): 
            for index, x, m in zip(indices, xs, ms):
                if len(m) == 0:
                    continue
                indices_known = list(set(range(x.shape[0])) - m) 
                indices_unknown = list(m)
                # Swap rows and columns s.t. the known values appear at the top
                # Keep both a forward and reversed mapping so we can swap them back afterwards.
                forward_mapping = indices_known + indices_unknown
                reversed_mapping = [0 for _ in forward_mapping]
                for i, j in enumerate(forward_mapping):
                    reversed_mapping[j] = i
                x_tmp = matrix_swap(x, forward_mapping)
                mu_tmp = matrix_swap(mu, forward_mapping)
                covariance_tmp = matrix_swap(covariance, forward_mapping)
                # Obtain values needed for computation
                a = len(indices_known)
                x_a = x_tmp[:a]
                x_m = x_tmp[a:]
                mu_a = mu_tmp[:a]
                mu_m = mu_tmp[a:]
                Sigma_aa = covariance_tmp[:a, :a]
                Sigma_am = covariance_tmp[:a, a:]
                Sigma_ma = covariance_tmp[a:, :a]
                Sigma_mm = covariance_tmp[a:, a:]
                # Find the new x_m. For now, just pick the mean value of the normal distribution.
                # Note that we use the pseudo-inverse to avoid rank issues.
                x_m = mu_m + Sigma_ma @ np.linalg.pinv(Sigma_aa) @ (x_a - mu_a)
                # Put the values back in our reference to x 
                x_tmp[a:] = x_m
                x_new = matrix_swap(x_tmp, reversed_mapping)
                x[:, :] = x_new[:, :]
                # Check that the only thing was changed was a missing value!
                assert(all(xs_input[index][j] == x[j] for j in range(x.shape[0]) if j not in m))
            # Update values
            X = np.hstack(xs)
            mu_prev = mu
            covariance_prev = covariance
            mu = np.mean(xs, axis = 0)
            covariance = np.cov(X)
            print(test_ctr)
            print(mu)
            # print(covariance)
            test_ctr += 1
        print(f'Loop ran {test_ctr} times!')
        for index, x, m in zip(indices, xs, ms):
            xs_derived[index] = x
            ys_derived[index] = y
        covariances.append(covariance)
    return xs_derived, ys_derived, covariances

def diabetes_dataset():
    X = []
    y = []
    with open('diabetes.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            x = [float(x) for x in row[:-1]]
            X.append(x)
            y.append(1 if int(row[-1]) == 1 else -1 )
    X = np.array(X)
    Y = np.array(y)

    # Testing the EM algo.
    xs_input = [x.reshape(-1, 1) for x in X]
    ys_input = [y for y in Y]
    ms_input = [set([]) for x in xs_input]
    estimate_missing_data(xs_input, ys_input, ms_input)

    # num_cv = 5
    # models = []
    # models.append( ("Regular SVM", SVC(gamma="auto")) )
    # for model_name, model in models:
    #     scores = cross_val_score(model, X, Y, cv = num_cv)
    #     print(f'For {model_name}, scores for {num_cv}-fold CV are: {scores}')



def main():
    diabetes_dataset()

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

if __name__ == "__main__":
    # main()
    # matrix_swap_test()
    # missing_data_test_small()
    missing_data_test_large()