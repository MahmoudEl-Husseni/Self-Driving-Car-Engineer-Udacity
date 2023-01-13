from utils import *
from sklearn.svm import SVC
from decision_tree import DecisionTree
from naive_bayes import GaussianNaiveBayes

# Generate data
X, y = make_classification(200)

# Gaussian Naive Bayes
NBclf = GaussianNaiveBayes()
NBclf.fit(X, y)
print(NBclf.mu, NBclf.sigma, NBclf.class_count)

# Suport Vector machine
svmclf = SVC(C=1, kernel='rbf')
svmclf.fit(X, y)

# Decision Tree
dtclf = DecisionTree(max_depth=15, min_samples_split=2)
dtclf.train(X, y)


# Plot
plt.figure(figsize=(20, 12))
ax = plt.subplot(131)
plot_classifier(X, y, NBclf, title="Naive Bayes Classifier", ax=ax)

ax = plt.subplot(132)
plot_classifier(X, y, svmclf, title='Support Vector Machine Classifier', ax=ax)

ax = plt.subplot(133)
plot_classifier(X, y, dtclf, title='Decision Tree Classifier', ax=ax)
plt.show()