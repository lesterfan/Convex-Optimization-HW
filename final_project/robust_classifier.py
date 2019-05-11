from sklearn.base import BaseEstimator, ClassifierMixin
from scipy.stats import norm
from scipy.linalg import sqrtm
import cvxpy as cp
import numpy as np

class RobustClassifier:
    def __init__(self, k = 0.5, ms = [], covariances = [], weight_val = 1):
        """

        Arguments:
            k (float) : Threshold probability of achieving the constriant as defined in the paper.
            ms (list(set(int))) : indices of missing variables
            covariances (list(np.array)) : the two calculated covariance matrices
            weight_val (float) : regularization parameter for robust SVM
        """
        self.k = k
        self.is_default_svm = False
        if len(ms) == 0:
            self.is_default_svm = True
        # inverse phi function of k (as defined in the paper)
        self.gamma = norm.ppf(k)
        self.ms = ms
        self.covariances = covariances
        self.weight_val = weight_val

        self.w_val = None
        self.b_val = None

    def fill_random_vals(self, num_vals, num_features):
        self.ms = [set([]) for _ in range(num_vals)]
        self.covariances = [np.random.random_sample( (num_features, num_features) ) for _ in range(2)]

    def fit(self, X, ys):
        """Fits the data using the robust classifier formulation
        presented in the paper.

        Arguments: 
            X (np.array) : data where each row is a feature vector
            ys (list(int)) : labels taking values in {-1, 1}
        """
        # Hacky way to set ms and covariances values if user didn't specify ms or covariances
        num_constraints, num_features = X.shape
        if self.is_default_svm:
            self.fill_random_vals(num_constraints, num_features)
        xs = [x.reshape(-1, 1) for x in X]
        dimensionality = xs[0].shape[0]
        # Formulate optimization variables and parameters
        w = cp.Variable(dimensionality)
        b = cp.Variable()
        u = cp.Variable(num_constraints)
        weight = cp.Parameter(nonneg = True)
        weight.value = self.weight_val
        # Formulate the optimization problem
        obj = cp.Minimize(cp.norm(u, p = 1))
        constraints = []
        for i in range(num_constraints):
            m = self.ms[i]
            # Nothing missing
            if len(m) == 0:
                constraints.append(
                    ys[i] * ( w.T * xs[i] + b ) >= 1 - u[i]
                )
            # Add in the alternate constraint from the paper
            else:
                # Create Sigma_i which is all 0 except for parts involving the msising variables
                Sigma_i = np.zeros( (num_features, num_features) , dtype = np.float64)
                calculated_covariance = self.covariances[0 if ys[i] == -1 else 1]
                for missing_index in m:
                    for c in range(num_features):
                        Sigma_i[missing_index][c] = calculated_covariance[missing_index][c]
                    for r in range(num_features):
                        Sigma_i[r][missing_index] = calculated_covariance[r][missing_index]
                # Add in new constraint
                matrix_sqrt, arg2 = sqrtm(Sigma_i, disp=False)
                # assert(arg2 != np.inf)
                # constraints.append(
                #          ys[i] * ( w.T * xs[i] + b ) >= 1 - u[i] + self.gamma * cp.norm(matrix_sqrt * w, p = 2)
                #       )
                # Hack, should change later
                if arg2 != np.inf:
                    constraints.append(
                        ys[i] * ( w.T * xs[i] + b ) >= 1 - u[i] + self.gamma * cp.norm(matrix_sqrt * w, p = 2)
                    )
                else:
                    constraints.append(
                        ys[i] * ( w.T * xs[i] + b ) >= 1 - u[i] + self.gamma * cp.norm( sqrtm(calculated_covariance) * w, p = 2)
                    )
                    # constraints.append(
                    #     ys[i] * ( w.T * xs[i] + b ) >= 1 - u[i]
                    # )
        constraints.append( u >= 0 )
        constraints.append( cp.norm(w, p = 2) <= weight )
        prob = cp.Problem(obj, constraints=constraints)
        # Solve the problem and store w, b, u
        prob.solve(solver = cp.ECOS, abstol = 1e-3, reltol = 1e-3)
        self.w_val = w.value
        self.b_val = b.value
        return self

    def predict(self, X): 
        """Predicts according to the values formed during fitting

        Arguments: 
            X (np.array) : data where each row is a feature vector

        Returns:
            ys (list(int)) : list of {-1, 1} values formed by prediction
        """
        assert(self.w_val is not None and self.b_val is not None)
        ys = []
        xs = [x.reshape(-1, 1) for x in X]
        for x in xs:
            if self.w_val.T @ x + self.b_val >= 0:
                ys.append(1)
            else:
                ys.append(-1)
        return ys

    def score(self, X, ys):
        """Evaluates the model on X according to true labels ys and returns
        the mean accuracy

        Arguments: 
            X (np.array) : data where each row is a feature vector
            ys (list(int)) : list of {-1, 1} values formed by prediction

        Returns:
            result(float) : mean accuracy achieved by this model
        """
        y_preds = self.predict(X)
        num_correct = sum([1 if y_pred == y else 0 for y_pred, y in zip(y_preds, ys)])
        result = num_correct / X.shape[0]
        return result