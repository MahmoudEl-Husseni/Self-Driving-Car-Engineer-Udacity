import numpy as np

# Utility functions
def determine_label(pool:np.ndarray):
    '''
    Determine label of pool of points.
    Args:
        pool: pool of points where each point is identified with its class value.
    Returns:
        label: label of pool of points.
    '''    
    return np.round(np.mean(pool))

def entropy(pool: np.ndarray) -> float: # returns entropy: sum(-pi*log(pi)) for in i all classes 
    entropy = 0
    for c in np.unique(pool):
        p = np.mean(pool==c) 
        entropy += - p * np.log2(p)        
    return entropy

def gini(pool : np.ndarray) -> float: # returns gini: sum(1-pi^2) for i in all classes 
    gini = 0
    for c in np.unique(pool):
        p = np.mean(pool==c)
        gini += 1 - np.power(p, 2)
    return gini


def information_gain(pool:np.ndarray, mask:np.ndarray, func=entropy,
                    F_parent=None) -> float:
    '''
    Calculate Information gain of feature: 
    Args: 
        pool: pool of points where each point is identified with its class value.
        mask: array masks each pool created after splitting main pool using our feature of interest.
        func: function used to determine impurity in pool (either "entropy" or "gini")
    Returns:
        info_gain: information gain value indicating amount of information provided from this feature. 
    '''
    if F_parent is None:
        F_parent = func(pool)

    F_childs_weighted = 0
    for _pool in np.unique(mask):
        F_child = func(pool[mask==_pool])
        weight = np.mean(mask==_pool)
        F_childs_weighted += weight * F_child

    info_gain = F_parent - F_childs_weighted
    return info_gain


def best_split(X, y, information_gain=information_gain, F_parent=None):
    '''
    Find best split for data X, y
    Args:
        X: features
        y: labels
    '''
    best_gain = 0
    best_feature = None
    best_mask = None
    best_thresh = None
    for i in range(X.shape[1]):
        feature = X[:, i]
        f = feature.copy()
        for thresh in np.unique(feature):
            f[feature>=thresh] = 1
            f[feature<=thresh] = 0
            f = f.astype(bool)
            gain = information_gain(y, f, F_parent=F_parent)
            if gain >= best_gain:
                best_gain = gain
                best_feature = i
                best_mask = f
                best_thresh = thresh
    return best_feature, best_mask, best_thresh
 
# ================================================================ # 
# Data Structures:
# ----------------
class Node:
    # Node class for decision tree
    def __init__(self, pool):
        
        self.pool = pool
        self.entropy = entropy(self.pool)
        self.gini = gini(self.pool)

        self.mask = None
        self.feature = None
        self.indices = None

        self.threshold = None

        self.label = None
        self.is_leaf = False

        self.tree = None # pointer to tree

    def calculate_entropy(self, pool=None):
        if pool is None:
            pool = self.pool
            self.entropy = entropy(pool)

        entropy = entropy(pool)
        return entropy
    
    def calculate_gini(self, pool=None):
        if pool is None:
            pool = self.pool
            self.gini = gini(pool)

        gini = gini(pool)
        return gini
    
    def find_best_split(self, features):
        best_feature, best_mask, best_thresh = best_split(features, self.pool, F_parent=self.entropy)
        self.mask = best_mask
        self.feature = best_feature
        self.threshold = best_thresh

        return best_feature, best_mask, best_thresh

    
class tree:
    def __init__(self, root:Node, leafs: list, best_feature:int) -> None:
        self.root = root
        self.leafs = leafs
        self.root.tree = self
        self.best_feature = best_feature
    


# ================================================================ #
# Decision Tree Algorithm:
# ------------------------

class DecisionTree:
    
    def __init__(self, max_depth=10, min_samples_split=2, verbose=False):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.root = None
        self.trees = []

        self.verbose = verbose
    

    def _train(self, X, y):
        '''
        Train decision tree
        Args:
            X: features
            y: labels
        '''
        self.trees = []
        depth = 1

        # create root node
        root = Node(y)
        root.indices = np.arange(X.shape[0])
        self.root = root
        best_feature, best_mask, _ = root.find_best_split(X)

        # create leafs of root node
        leafs = []
        for c in np.unique(best_mask):
            pool = y[best_mask==c]
            node = Node(pool)
            node.indices = root.indices[best_mask==c]
            leafs.append(node)

        # store trees
        self.trees.append(tree(root, leafs, best_feature))

        
        while(depth <= self.max_depth):
            if self.verbose:
                print("Decision Tree depth: ", depth)
                print("number of roots", len(leafs))
            
            roots = leafs.copy()
            leafs = []
            n_roots = len(roots)
            
            if n_roots == 0:
                break

            n_trees = 0
            # Create trees for each leaf
            for root in roots:
                best_feature, best_mask, _ = root.find_best_split(X[root.indices, :])

                if best_mask is None:
                    root.is_leaf = True
                    root.label = determine_label(root.pool)
                    continue

                if (best_mask.sum() == 0) or (best_mask.sum() == len(best_mask)):
                    best_mask = best_mask if best_mask.sum() == len(best_mask) else ~best_mask
                    pool = y[root.indices][best_mask]
                    node = Node(pool)
                    node.indices = root.indices[best_mask]
                    node.is_leaf = True
                    node.label = determine_label(pool)
                    self.trees.append(tree(root, [node], best_feature))
                    continue
                
                for c in np.unique(best_mask):
                    pool = y[root.indices][best_mask==c]
                    node = Node(pool)
                    node.indices = root.indices[best_mask==c]
                    leafs.append(node)


                self.trees.append(tree(root, leafs, best_feature))
                n_trees += 1


            # check if leafs can be splitted further
            for _tree in self.trees[-n_trees:]:
                for leaf in _tree.leafs:
                    if leaf.entropy == 0:
                        try: 
                            leafs.remove(leaf)
                        finally:
                            leaf.is_leaf = True
                            leaf.label = determine_label(leaf.pool)
                            continue

                    if len(leaf.pool) < self.min_samples_split:
                        leafs.remove(leaf)
                        leaf.is_leaf = True
                        leaf.label = determine_label(leaf.pool)
                        continue

                if len(_tree.leafs) < 2:
                    _tree.root.is_leaf = True
                    _tree.root.label = determine_label(_tree.root.pool)
                    self.trees.remove(_tree)
            depth += 1


    def predict(self, X):
        labels = []
        
        for x in X:
            node = self.root
            tree = self.trees[0]
            while(True):
                feature = tree.root.feature
                if tree.root.is_leaf:
                    labels.append(tree.root.label)
                    break 

                if x[feature] <= tree.root.threshold:
                    node = tree.leafs[0]
                    tree = node.tree

                else:
                    node = tree.leafs[1]
                    tree = node.tree

                if node.is_leaf:
                    labels.append(node.label)
                    break
        return np.array(labels)

    def train(self, X, y):
        '''
        Train decision tree
        Args:
            X: features
            y: labels
        '''
        self.root = self._train(X, y)



# test
if __name__=="__main__":
    """
    Gender  Height  Weight  Index
    0    Male     174      96      4
    1    Male     189      87      2
    2  Female     185     110      4
    3  Female     195     104      3
    4    Male     149      61      3
    """
    data = np.array([[1, 174, 96, 1], 
                    [1, 189, 87, 0], 
                    [0, 185, 110, 1], 
                    [0, 195, 104, 0], 
                    [1, 149, 61, 0]])

    dt = DecisionTree(verbose=True)
    dt.train(data[:, :-1], data[:, -1])
    print(len(dt.trees))
    print(dt.predict([[1, 174, 96]]))