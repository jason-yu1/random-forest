import numpy as np
from sklearn.utils import resample
from sklearn.metrics import r2_score, accuracy_score
import dtree
import copy

class RandomForest621:
    def __init__(self, n_estimators=10, oob_score=False):
        self.n_estimators = n_estimators
        self.nunique = 0
        self.oob_score = oob_score
        self.oob_score_ = np.nan

    def fit(self, X, y):
        """
        Given an (X, y) training set, fit all n_estimators trees to different,
        bootstrapped versions of the training data. Keep track of the indexes of
        the OOB records for each tree. After fitting all of the trees in the forest,
        compute the OOB validation score estimate and store as self.oob_score_, to
        mimic sklearn.
        """
        trees = []
        for i in range(self.n_estimators):
            # bootstrap: sample with replacements
            n = len(y)
            idx = np.random.randint(0, n, size=n)
            X_train = X[idx]
            y_train = y[idx]

            new_tree = copy.deepcopy(self.tree)
            trees.append(new_tree(min_samples_leaf=self.min_samples_leaf))
            trees[i].fit(X_train, y_train, self.max_features)

            # get OOB samples
            mask = np.ones(n, dtype=bool)
            mask[idx] = False
            X_test = X[mask]
            y_test = y[mask]
            trees[i].oob_idxs = X_test
            trees[i].oob_idys = y_test
            trees[i].oob_ids = mask

        self.trees = trees

        if self.oob_score:
            self.oob_score_ = self.compute_oob_score(X, y)


class RandomForestRegressor621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False, tree=dtree.RegressionTree621):
        super().__init__(n_estimators=n_estimators, oob_score=oob_score)
        self.trees = None
        self.oob_score_ = None
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = tree

    def compute_oob_score(self, X, y):
        oob_counts = np.array([0 for i in range(len(X))], dtype=float)
        oob_preds = np.array([0 for i in range(len(X))], dtype=float)
        for tree in self.trees:
            # num samples in leaf reached by each OOB X
            leaf_sizes = [tree.root.leaf(idx).n for idx in tree.oob_idxs]
            oob_preds[tree.oob_ids] += leaf_sizes * tree.predict(tree.oob_idxs)
            oob_counts[tree.oob_ids] += leaf_sizes

        oob_avg_preds = oob_preds[oob_counts > 0] / oob_counts[oob_counts > 0]

        return r2_score(y[oob_counts > 0], oob_avg_preds)

    def predict(self, X_test) -> np.ndarray:
        """
        Given a 2D nxp array with one or more records, compute the weighted average
        prediction from all trees in this forest. Weight each trees prediction by
        the number of samples in the leaf making that prediction. Return a 1D vector
        with the predictions for each input record of X_test.
        """
        preds = np.zeros(X_test.shape[0])
        for i, X in enumerate(X_test):
            leaves = [tree.root.leaf(X) for tree in self.trees]
            nobs = np.sum([leaf.n for leaf in leaves])
            ysum = np.sum([leaf.prediction * leaf.n for leaf in leaves])

            preds[i] = (1/nobs)*ysum

        return preds

    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records ,
        collect the prediction for each record and then compute R^2 on that and y_test
        """
        y_pred = self.predict(X_test)
        return r2_score(y_test, y_pred)


class RandomForestClassifier621(RandomForest621):
    def __init__(self, n_estimators=10, min_samples_leaf=3, max_features=0.3, oob_score=False, tree=dtree.ClassifierTree621):
        super().__init__(n_estimators=n_estimators, oob_score=oob_score)
        self.trees = None
        self.oob_score_ = None
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.tree = tree

    def compute_oob_score(self, X, y):
        oob_counts = np.array([0 for i in range(len(X))], dtype=float)
        # create 2D matrix tracking vote counts per class for each obs
        oob_preds = np.zeros(shape=(X.shape[0], len(np.unique(y))))

        for tree in self.trees:
            # num samples in leaf reached by each OOB X
            leaf_sizes = [tree.root.leaf(idx).n for idx in tree.oob_idxs]
            tpred = tree.predict(tree.oob_idxs).astype(int)
            oob_preds[tree.oob_ids, tpred] += leaf_sizes
            oob_counts[tree.oob_ids] += 1

        nonzero_idx = np.where(oob_counts > 0)
        oob_votes = np.zeros(shape=oob_counts.shape[0])
        for i in nonzero_idx:
            oob_votes[i] = np.argmax(oob_preds[i], axis=1)

        return accuracy_score(y[oob_counts > 0], oob_votes[oob_counts > 0])

    def predict(self, X_test) -> np.ndarray:
        preds = np.zeros(X_test.shape[0])
        for i, X in enumerate(X_test):
            counts = np.zeros(999)
            for tree in self.trees:
                leaf = tree.root.leaf(X)
                counts[leaf.prediction] += 1
            preds[i] = np.argmax(counts, axis=0)

        return preds


    def score(self, X_test, y_test) -> float:
        """
        Given a 2D nxp X_test array and 1D nx1 y_test array with one or more records,
        collect the predicted class for each record and then compute accuracy between
        that and y_test.
        """
        y_pred = self.predict(X_test)
        return accuracy_score(y_test, y_pred)









