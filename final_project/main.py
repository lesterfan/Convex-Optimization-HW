from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.datasets import make_classification
from sklearn.datasets import load_breast_cancer
from sklearn.svm import SVC
from sklearn import preprocessing
from robust_classifier import RobustClassifier
import matplotlib.pyplot as plt
import numpy as np
import itertools
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

def estimate_missing_data(xs_input, ys_input, ms_input, mu_rtol = 1e-2, mu_atol = 1e-2, cov_atol = 1e-1, cov_rtol = 1e-1):
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
        # X is the matrix where the cols are the feature vectors
        X = np.hstack(xs)
        # Can either pick mu and covariance based off initial data or randomly
        mu = np.mean(xs, axis = 0, dtype = np.float64)
        covariance = np.cov(X)
        # mu = np.random.random_sample( xs[0].shape )
        # covariance = np.random.random_sample( (xs[0].shape[0], xs[0].shape[0]) )
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
            # print(mu)
            # print(covariance)
            test_ctr += 1
        print(f'Loop ran {test_ctr} times!')
        for index, x, m in zip(indices, xs, ms):
            xs_derived[index] = x
            ys_derived[index] = y
        covariances.append(covariance)
    return xs_derived, ys_derived, covariances

def estimate_missing_matrix(X, ys, ms, **kwargs):
    """Wrapper for estimate_missing_data which accepts the matrix
    X where each row is a feature vector. Returns the data in the same
    format.

    Arguments: 
        X (np.array) : array where each row is a feature vector
        ys (list(int)) : labels taking values in {-1, 1}
        ms (list(set(int))) : indices of missing values

    Returns:
        X_derived (np.array) : filled array where each row is a feature vector
        ys_derived (list of ints which take values from {-1, 1}) : the corresponding classes
        covariances (list of np.arrays of shape (n, n)) : covariance matrices, one for each
                                                              class (so two in total)
    """
    xs = [x.reshape(-1, 1) for x in X]
    xs_derived, ys_derived, covariances = estimate_missing_data(xs, ys, ms, **kwargs)
    X_derived = np.hstack(xs_derived).T
    return X_derived, ys_derived, covariances

def generate_ms(X, p):
    """Generates ms (as defined in estimate_missing_data) by removing
    each element of X with a Uniform(p) probability

    Arguments:
        X (np.array) : array where each row is a feature vector
        p (float) : the probability
    
    Returns:
        ms (list(set(int))) : missing values for each of the data
    """
    assert(0 <= p and p <= 1)
    R, C = X.shape
    num_samples = int(R * C * p)
    omissions = random.sample( list(itertools.product(range(R), range(C))) , num_samples)
    ms = [ set([]) for r in range(R) ]
    # Omit element c from the rth feature vector
    for r, c in omissions:
        ms[r].add(c)
    return ms

def k_fold_cv(X, ys, ms, cv=5, threshold_prob = 0.5, case = 1):
    kf = KFold(n_splits=cv)
    scores = []
    for train, test in kf.split(X):
        # Get the training and test splits
        X_train = np.vstack( [ X[i] for i in train ] )
        ys_train = [ys[i] for i in train ]
        ms_train = [ms[i] for i in train ]
        X_test = np.vstack( [ X[i] for i in test ] )
        ys_test = [ys[i] for i in test ]
        # Regular SVM
        if case == 1:
            model = RobustClassifier()
            model.fit(X_train, ys_train)
            scores.append(model.score(X_test, ys_test))
        # Regular SVM, but using the calculated values as fact
        elif case == 2:
            X_derived, ys_derived, covariances = estimate_missing_matrix(X_train, ys_train, ms_train)
            model = RobustClassifier()
            model.fit(X_derived, ys_train)
            scores.append(model.score(X_test, ys_test))
        # Robust SVM
        elif case == 3:
            X_derived, ys_derived, covariances = estimate_missing_matrix(X_train, ys_train, ms_train)
            model = RobustClassifier(k = threshold_prob, ms = ms_train, covariances = covariances)
            model.fit(X_derived, ys_train)
            scores.append(model.score(X_test, ys_test))
    return scores

def evaluate(X, ys, p, num_cv, threshold_prob, **kwargs):
    """Evaluates performance on the given input data using {num_cv}-fold cross validation

    Arguments: 
        X (np.array) : data where each row is a feature vector
        ys (list(int)) : labels taking values in {-1, 1}
        p (float) : between 0 and 1, probability for a single value to be removed
        num_cv (int) : number of cross validation folds to use

    Returns:
        result (list(int)) : avg scores across each model
    """
    ms = generate_ms(X, p)

    # 0. Regular SVM with all of the inputs (using RobustClassifier with no inputs to keep the same regularization parameter)
    # model_0 = RobustClassifier()
    # scores_0 = k_fold_cv(model_0, X, ys, cv = num_cv)
    # print(f'For regular SVM using {X.shape[0]} samples, the average score for {num_cv}-fold CV are: {np.mean(scores_0)}')
    scores_0 = [0]

    # 1. Regular SVM with p samples randomly removed 
    turncated_indices = random.sample(list(range(X.shape[0])), int(X.shape[0] * (1 - p)))
    X_1 = np.vstack([X[index] for index in turncated_indices])
    ys_1 = [ys[index] for index in turncated_indices]
    scores_1 = k_fold_cv(X_1, ys_1, ms, cv = num_cv, threshold_prob = threshold_prob, case = 1)
    print(f'For SVM now with {X_1.shape[0]} samples, the average score for {num_cv}-fold CV are: {np.mean(scores_1)}')

    # 2. Regular SVM using only the values deduced through EM
    scores_2 = k_fold_cv(X, ys, ms, cv = num_cv, threshold_prob = threshold_prob, case = 1)
    print(f'For SVM cnsidering only the mean from EM, the average score for {num_cv}-fold CV are: {np.mean(scores_2)}')

    # 3. Robust SVM now considering missing values and their covariances!
    scores_3 = k_fold_cv(X, ys, ms, cv = num_cv, threshold_prob = threshold_prob, case = 2)
    print(f'For robust SVM, the average score for {num_cv}-fold CV are: {np.mean(scores_3)}')

    return [ np.mean(scores) for scores in [scores_0, scores_1, scores_2, scores_3] ]

def diabetes_dataset(p, num_cv, n_samples, threshold_prob):
    X = []
    y = []
    with open('diabetes.csv') as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if i == 0:
                continue
            if i > n_samples:
                break
            x = [float(x) for x in row[:-1]]
            X.append(x)
            y.append(1 if int(row[-1]) == 1 else -1 )
    X = np.array(X, dtype = np.float64)
    X_scaled = preprocessing.scale(X)
    ys = np.array(y, dtype = np.float64)
    return evaluate(X_scaled, ys, p, num_cv, threshold_prob)

def generic_dataset(p, num_cv, n_samples, n_features, n_informative, threshold_prob):
    X, ys = make_classification(
        n_samples=n_samples, 
        n_features=n_features, 
        n_informative=n_informative,
        n_redundant=0
        )
    X = X.astype(np.float64)
    X_scaled = preprocessing.scale(X)
    ys = [1 if y == 1 else -1 for y in ys]
    return evaluate(X_scaled, ys, p, num_cv, threshold_prob)

def breast_cancer_dataset(p, num_cv, threshold_prob):
    X, ys = load_breast_cancer(return_X_y=True)
    X = X.astype(np.float64)
    X_scaled = preprocessing.scale(X)
    ys = [1 if y == 1 else -1 for y in ys]
    return evaluate(X_scaled, ys, p, num_cv, threshold_prob)

if __name__ == "__main__":
    omission_probs = np.linspace(0.05, 0.95, num=10)
    reg_scores, removed_scores, only_means_scores, robust_scores = [], [], [], []
    threshold_prob = 0.85
    title = "Breast Cancer Results"
    for i, p in enumerate(omission_probs):
        print(f'Trial {i} out of {len(omission_probs) - 1}. p = {p}.')
        # Uncomment datasets to obtain results for that particular dataset
        # reg_score, removed_score, only_means_score, robust_score = diabetes_dataset(p, 5, 1000, threshold_prob)
        reg_score, removed_score, only_means_score, robust_score = breast_cancer_dataset(p, 5, threshold_prob)
        # reg_score, removed_score, only_means_score, robust_score = generic_dataset(p, 5, 400, 20, 20, threshold_prob)
        # reg_scores.append(reg_score)
        removed_scores.append(removed_score)
        only_means_scores.append(only_means_score)
        robust_scores.append(robust_score)
    # p1 = plt.plot(omission_probs, reg_scores, 'xb-', label = "Regular SVM")
    p2 = plt.plot(omission_probs, removed_scores, 'xr-', label = "SVM with proportion p points removed")
    p3 = plt.plot(omission_probs, only_means_scores, 'xk-', label = "SVM with only using mean values from EM")
    p4 = plt.plot(omission_probs, robust_scores, 'xg-', label = "Robust SVM")
    plt.title(title)
    plt.xlabel("Fraction of Input Data Removed (p)")
    plt.ylabel("Model Accuracy")
    plt.legend()
    print("Done!")
    plt.show()