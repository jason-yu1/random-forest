import numpy as np
from sklearn.metrics import r2_score, accuracy_score
from scipy import stats

class DecisionNode:
    def __init__(self, col, split, lchild, rchild):
        self.col = col
        self.split = split
        self.lchild = lchild
        self.rchild = rchild


    def predict(self, x_test):
        # Make decision based upon x_test[col] and split
        if x_test[self.col] <= self.split:
            return self.lchild.predict(x_test)
        return self.rchild.predict(x_test)


    def leaf(self, x_test):
        """
        Given a single test record, x_test, return the leaf node reached by running
        it down the tree starting at this node.  This is just like prediction,
        except we return the decision tree leaf rather than the prediction from that leaf.
        """

        if x_test[self.col] <= self.split:
            return self.lchild.leaf(x_test)

        return self.rchild.leaf(x_test)



class LeafNode:
    def __init__(self, y, prediction):
        "Create leaf node from y values and prediction; prediction is mean(y) or mode(y)"
        self.n = len(y)
        self.prediction = prediction

    def predict(self, x_test):
        # return prediction passed to constructor of LeafNode
        return self.prediction

    def leaf(self, x_test):
        return self



class DecisionTree621:
    def __init__(self, min_samples_leaf=1, loss=None):
        self.min_samples_leaf = min_samples_leaf
        self.loss = loss  # loss function; either np.std or gini
        # self.oob_idxs = None
        # self.oob_idys = None

    def fit(self, X, y, max_features):
        """
        Create a decision tree fit to (X,y) and save as self.root, the root of
        our decision tree, for either a classifier or regressor.  Leaf nodes for classifiers
        predict the most common class (the mode) and regressors predict the average y
        for samples in that leaf.

        This function is a wrapper around fit_() that just stores the tree in self.root.
        """
        self.root = self.fit_(X, y, max_features)

    def bestsplit(self, X, y, loss, max_features):
        best = (-1, -1, loss(y))
        max_vars = int(round(X.shape[1] * max_features))
        vars = np.random.choice(range(X.shape[1]), size=max_vars, replace=False)
        for col in vars:
            # randomly pick k values from col
            candidates = np.random.choice(X[:, col], min(11, X.shape[0]), replace=False)

            for split in candidates:
                yl = y[X[:, col] <= split]
                yr = y[X[:, col] > split]
                if (len(yl) < self.min_samples_leaf) or (len(yr) < self.min_samples_leaf):
                    continue

                l = (len(yl) * loss(yl) + len(yr) * loss(yr))/len(y)

                if l == 0:
                    return col, split
                if l < best[2]:
                    best = (col, split, l)

        return best[0], best[1]

    def fit_(self, X, y, max_features):
        """
        Recursively create and return a decision tree fit to (X,y) for
        either a classifier or regressor.  This function should call self.create_leaf(X,y)
        to create the appropriate leaf node, which will invoke either
        RegressionTree621.create_leaf() or ClassifierTree621. create_leaf() depending
        on the type of self.

        This function is not part of the class "interface" and is for internal use, but it
        embodies the decision tree fitting algorithm.

        (Make sure to call fit_() not fit() recursively.)
        """
        if len(X) < self.min_samples_leaf:
            return LeafNode(y)
        col, split = self.bestsplit(X, y, self.loss, max_features)
        if col == -1:
            return self.create_leaf(y)
        lchild = self.fit_(X[X[:, col] <= split], y[X[:, col] <= split], max_features)
        rchild = self.fit_(X[X[:, col] > split], y[X[:, col] > split], max_features)

        return DecisionNode(col, split, lchild, rchild)

    def predict(self, X_test):
        """
        Make a prediction for each record in X_test and return as array.
        This method is inherited by RegressionTree621 and ClassifierTree621 and
        works for both without modification!
        """
        pred = np.zeros(X_test.shape[0])
        for i, record in enumerate(X_test):
            pred[i] = self.root.predict(record)
        return pred


class RegressionTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=np.std)
    def score(self, X_test, y_test):
        "Return the R^2 of y_test vs predictions for each record in X_test"
        return r2_score(y_test, self.predict(X_test))


    def create_leaf(self, y):
        """
        Return a new LeafNode for regression, passing y and mean(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, np.mean(y))


class ClassifierTree621(DecisionTree621):
    def __init__(self, min_samples_leaf=1):
        super().__init__(min_samples_leaf, loss=gini)
    def score(self, X_test, y_test):
        "Return the accuracy_score() of y_test vs predictions for each record in X_test"
        return accuracy_score(y_test, self.predict(X_test))


    def create_leaf(self, y):
        """
        Return a new LeafNode for classification, passing y and mode(y) to
        the LeafNode constructor.
        """
        return LeafNode(y, stats.mode(y)[0])


def gini(y):
    g = 0
    for i in set(y):
        g += ((len(y[y==i]))/len(y))*(1-((len(y[y==i]))/len(y)))

    return g









