import numpy as np
from matplotlib import pyplot as plt


# Make Data
def make_classification(n_samples:int = 200, threshold:float = 0.5):
    x1 = np.random.rand(n_samples)
    x2 = np.random.rand(n_samples)
    
    X = np.vstack([x1, x2]).T
    y = np.zeros(n_samples)
    y[(x1 * x2 + 0.3) > threshold] = 1

    return X, y


# Plot Data
def plot_classifier(X, y, clf, title="Classifier", ax=None):
    if ax is None:
        ax = plt.gca()
    ax.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='RdBu', vmin=-.2*X.min(), vmax=1.2*X.max())
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_title(title)
    ax.grid()
    ax.legend()

    x1 = np.linspace(X.min(), X.max(), 100)
    x2 = np.linspace(X.min(), X.max(), 100)
    X1, X2 = np.meshgrid(x1, x2)
    X_grid = np.vstack([X1.ravel(), X2.ravel()]).T
    y_grid = clf.predict(X_grid).reshape(X1.shape)

    ax.pcolormesh(X1, X2, y_grid, alpha=0.2, cmap='RdBu', vmin=-.2*X.min(), vmax=1.2*X.max())

    return ax