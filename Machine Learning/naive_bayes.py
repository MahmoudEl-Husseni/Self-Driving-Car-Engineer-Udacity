import numpy as np


class GaussianNaiveBayes:

    def __init__(self) -> None:

        self.mu = None
        self.sigma = None
        self.class_count = None

    def split_by_class(self, X, y):
        """Split X and y by class.

        Args:
            X (np.ndarray): X
            y (np.ndarray): y

        Returns:
            np.ndarray: X by class
            np.ndarray: y by class
        """
        X_per_class = []
        classes = []
        for i in np.unique(y):
            X_per_class.append(X[y == i])
            classes.append(i)
        
        self.class_count = len(classes)
        return X_per_class, classes


    def fit(self, X, y):
        """Calculate parameters mu and sigma for each class.

        Args:
            X (np.ndarray): X
            y (np.ndarray): y
        """
        X_by_class, classes = self.split_by_class(X, y)
        
        self.mu = []
        self.sigma = []
        for i in range(self.class_count):
            self.mu.append(np.mean(X_by_class[i], axis=0))
            self.sigma.append(np.std(X_by_class[i], axis=0))


    def calc_prob(self, x, mu, sigma):
        """Calculate probability of x given mu and sigma.

        Args:
            x (float): x
            mu (float): mu
            sigma (float): sigma

        Returns:
            float: probability
        """
        return np.exp(-((x - mu) ** 2) / (2 * sigma ** 2)) / (sigma * np.sqrt(2 * np.pi))

    def predict(self, X):
        """Predict y given X.

        Args:
            X (np.ndarray): X

        Returns:
            np.ndarray: y
        """
        y = []
        for x in X: # X = [[x1, x2], [x1, x2], ...
            prob = []
            for i in range(self.class_count):
                prob.append(np.prod(self.calc_prob(x, self.mu[i], self.sigma[i])))
            y.append(np.argmax(prob))
        return np.array(y)