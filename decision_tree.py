# decision_tree.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# personal and educational purposes provided that (1) you do not distribute
# or publish solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UT Dallas, including a link to http://cs.utdallas.edu.
#
# This file is part of Programming Assignment 2 for CS6375: Machine Learning.
# Gautam Kunapuli (gautam.kunapuli@utdallas.edu)
# Sriraam Natarajan (sriraam.natarajan@utdallas.edu),
#
# Tahmid Imran
#
# INSTRUCTIONS:
# ------------
# 1. This file contains a skeleton for implementing the ID3 algorithm for
# Decision Trees. Insert your code into the various functions that have the
# comment "INSERT YOUR CODE HERE".
#
# 2. Do NOT modify the classes or functions that have the comment "DO NOT
# MODIFY THIS FUNCTION".
#
# 3. Do not modify the function headers for ANY of the functions.
#
# 4. You may add any other helper functions you feel you may need to print,
# visualize, test, or save the data and results. However, you MAY NOT utilize
# the package scikit-learn OR ANY OTHER machine learning package in THIS file.

import numpy as np
import math
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
import graphviz
from sklearn.metrics import plot_confusion_matrix
import random
from sklearn.ensemble import BaggingClassifier, AdaBoostClassifier


def partition(x):
    """
    Partition the column vector x into subsets indexed by its unique values (v1, ... vk)

    Returns a dictionary of the form
    { v1: indices of x == v1,
      v2: indices of x == v2,
      ...
      vk: indices of x == vk }, where [v1, ... vk] are all the unique values in the vector z.
    """

    # INSERT YOUR CODE HERE
    v = {}
    for i in range(len(x)):
        if x[i] not in v:
            v[x[i]] = []
        v[x[i]].append(i)
    return v


def entropy(y, weights):
    """
    Compute the entropy of a vector y by considering the counts of the unique values (v1, ... vk), in z

    Returns the entropy of z: H(z) = p(z=v1) log2(p(z=v1)) + ... + p(z=vk) log2(p(z=vk))
    """

    # INSERT YOUR CODE HERE
    part = partition(y)
    totalCount = len(y)
    Hz = 0.0
    weights = np.array(weights)
    for key in part:
        vCount = len(part.get(key))
        vProb = vCount/totalCount
        Hz += -vProb * math.log(vProb, 2)
    return Hz


def mutual_information(x, y, weights):
    """
    Compute the mutual information between a data column (x) and the labels (y). The data column is a single attribute
    over all the examples (n x 1). Mutual information is the difference between the entropy BEFORE the split set, and
    the weighted-average entropy of EACH possible split.

    Returns the mutual information: I(x, y) = H(y) - H(y | x)
    """

    # INSERT YOUR CODE HERE
    Hy = entropy(y, weights)
    xVal, xCounts = np.unique(x, return_counts=True)
    Hyx = 0.0
    for i in range(len(xVal)):
        prob = xCounts[i]/len(x)
        Hyx += sum(weights[x == xVal[i]])*prob*entropy(y[x == xVal[i]], weights)
    return Hy-Hyx


def id3(x, y, attribute_value_pairs=None, depth=0, max_depth=5, weights=None):
    """
    Implements the classical ID3 algorithm given training data (x), training labels (y) and an array of
    attribute-value pairs to consider. This is a recursive algorithm that depends on three termination conditions
        1. If the entire set of labels (y) is pure (all y = only 0 or only 1), then return that label
        2. If the set of attribute-value pairs is empty (there is nothing to split on), then return the most common
           value of y (majority label)
        3. If the max_depth is reached (pre-pruning bias), then return the most common value of y (majority label)
    Otherwise the algorithm selects the next best attribute-value pair using INFORMATION GAIN as the splitting criterion
    and partitions the data set based on the values of that attribute before the next recursive call to ID3.

    The tree we learn is a BINARY tree, which means that every node has only two branches. The splitting criterion has
    to be chosen from among all possible attribute-value pairs. That is, for a problem with two features/attributes x1
    (taking values a, b, c) and x2 (taking values d, e), the initial attribute value pair list is a list of all pairs of
    attributes with their corresponding values:
    [(x1, a),
     (x1, b),
     (x1, c),
     (x2, d),
     (x2, e)]
     If we select (x2, d) as the best attribute-value pair, then the new decision node becomes: [ (x2 == d)? ] and
     the attribute-value pair (x2, d) is removed from the list of attribute_value_pairs.

    The tree is stored as a nested dictionary, where each entry is of the form
                    (attribute_index, attribute_value, True/False): subtree
    * The (attribute_index, attribute_value) determines the splitting criterion of the current node. For example, (4, 2)
    indicates that we test if (x4 == 2) at the current node.
    * The subtree itself can be nested dictionary, or a single label (leaf node).
    * Leaf nodes are (majority) class labels

    Returns a decision tree represented as a nested dictionary, for example
    {(4, 1, False):
        {(0, 1, False):
            {(1, 1, False): 1,
             (1, 1, True): 0},
         (0, 1, True):
            {(1, 1, False): 0,
             (1, 1, True): 1}},
     (4, 1, True): 1}
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if len(set(y)) == 1:
        return y[0]

    if attribute_value_pairs is None:
        attribute_value_pairs = []
        for i in range(len(x[0])):
            uniqueVals = np.unique(x[:, i])
            for j in range(len((uniqueVals))):
                attribute_value_pairs.append((i, uniqueVals[j]))

    if weights is None:
        weights = np.array([(1/len(x)) for _ in x])

    if len(attribute_value_pairs) == 0 or depth == max_depth:
        # Finds all unique elements and their positions
        unique, pos = np.unique(y, return_inverse=True)
        counts = np.bincount(pos)  # Count the number of each unique element
        maxpos = counts.argmax()
        return unique[maxpos]

    node = {}

    maxInfo = 0
    maxVariable = None
    for (k, v) in attribute_value_pairs:
        info = mutual_information(np.array(x[:, k] == v), y, weights)
        if info > maxInfo:
            maxInfo = info
            maxVariable = (k, v)

    xTrue = x[(x[:, maxVariable[0]] == maxVariable[1])]
    xFalse = x[(x[:, maxVariable[0]] != maxVariable[1])]
    yTrue = y[(x[:, maxVariable[0]] == maxVariable[1])]
    yFalse = y[(x[:, maxVariable[0]] != maxVariable[1])]
    weightsTrue = weights[(x[:, maxVariable[0]] == maxVariable[1])]
    weightsFalse = weights[(x[:, maxVariable[0]] != maxVariable[1])]
    node[(maxVariable[0], maxVariable[1], False)] = id3(xFalse, yFalse,
                                                        attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth, weights=weightsFalse)
    node[(maxVariable[0], maxVariable[1], True)] = id3(xTrue, yTrue,
                                                       attribute_value_pairs=attribute_value_pairs, depth=depth+1, max_depth=max_depth, weights=weightsTrue)

    return node


def bagging(x, y, max_depth, num_trees):
    hypotheses = []
    alpha = -1
    for _ in range(0, num_trees):
        # randomly select 63% of data
        idx = np.random.choice(
            np.arange(len(x)), int(.63*len(x)), replace=False)
        x_sample = x[idx]
        y_sample = y[idx]
        # generate decision tree learned by this portion of data
        hypotheses.append(
            (alpha, id3(x_sample, y_sample, max_depth=max_depth)))
    return hypotheses


def boosting(x, y, max_depth, num_stumps):
    hypotheses = []
    d = np.array([(1/len(x)) for _ in x])  # sample weights
    for _ in range(num_stumps):
        tree = id3(x, y, max_depth=max_depth, weights=d)

        y_pred = [predict_example_learner(xi, tree) for xi in x]
        e = compute_error_weights(y, y_pred, d)

        alpha_i = -1
        if e > 0 and e < 1:
            alpha_i = .5 * math.log((1 - e)/e)
        for di in range(len(d)):  # update weights
            d[di] = d[di] * math.exp(alpha_i * 1 if y_pred[di] != y[di] else -1)
        total = sum(d)
        for di in range(len(d)):  # normalize weights
            d[di] /= total
        hypotheses.append((alpha_i, tree))
    return hypotheses


def compute_error_weights(y_true, y_pred, weights):
    e = 0
    for i in range(len(y_true)):
        e += weights[i] if y_true[i] != y_pred[i] else 0
    return e


def predict_example_learner(x, tree):
    """
    Predicts the classification label for a single example x using tree by recursively descending the tree until
    a label/leaf node is reached.

    Returns the predicted label of x according to tree
    """

    # INSERT YOUR CODE HERE. NOTE: THIS IS A RECURSIVE FUNCTION.
    if isinstance(tree, (int, np.int64, np.int32, np.str_)):
        return tree
    for key in tree.keys():
        present = x[key[0]] == key[1]
        predicted_label = predict_example_learner(
            x, tree[(key[0], key[1], present)])
        break
    return predicted_label


def predict_example(x, h_ens):
    '''
    h_ens is an ensemble of weighted hypotheses. The ensemble is represented as an array of pairs [(alpha_i, h_i)], where each hypothesis and weight are represented by the pair: (alpha_i, h_i)
    '''
    weighted_vote = 0
    for pair in h_ens:
        alpha_i = pair[0]
        h_i = pair[1]
        prediction = int(predict_example_learner(x, h_i))
        weighted_vote += -alpha_i * (1 if prediction == 1 else -1)
    if weighted_vote > 0:
        return 1
    elif weighted_vote < 0:
        return 0
    return random.randint(0, 1)


def compute_error(y_true, y_pred):
    """
    Computes the average error between the true labels (y_true) and the predicted labels (y_pred)

    Returns the error = (1/n) * sum(y_true != y_pred)
    """

    # INSERT YOUR CODE HERE
    count = 0
    for i in range(len(y_true)):
        count += 0 if y_true[i] == y_pred[i] else 1
    return count / len(y_true)


def visualize(tree, depth=0):
    """
    Pretty prints (kinda ugly, but hey, it's better than nothing) the decision tree to the console. Use print(tree) to
    print the raw nested dictionary representation.
    DO NOT MODIFY THIS FUNCTION!
    """

    if depth == 0:
        print('TREE')

    for index, split_criterion in enumerate(tree):
        sub_trees = tree[split_criterion]

        # Print the current node: split criterion
        print('|\t' * depth, end='')
        print(
            '+-- [SPLIT: x{0} = {1}]'.format(split_criterion[0], split_criterion[1]))

        # Print the children
        if type(sub_trees) is dict:
            visualize(sub_trees, depth + 1)
        else:
            print('|\t' * (depth + 1), end='')
            print('+-- [LABEL = {0}]'.format(sub_trees))


def confusion_matrix(h_ens, Xtst, ytst, scikit=False):
    tp = 0
    fp = 0
    tn = 0
    fn = 0
    if not scikit:
        y_pred = [predict_example(x, h_ens) for x in Xtst]
    else:
        y_pred = h_ens
    for i in range(len(ytst)):
        if ytst[i] == 1 and y_pred[i] == 1:
            tp += 1
        elif ytst[i] == 1 and y_pred[i] == 0:
            fn += 1
        elif ytst[i] == 0 and y_pred[i] == 1:
            fp += 1
        elif ytst[i] == 0 and y_pred[i] == 0:
            tn += 1
    l = [[tp, fn], [fp, tn]]
    print(l)
    print('                   Confusion Matrix')
    print('                      Prediction')
    print('                 Positive   Negative')
    print('               |----------|----------|')
    print('               |          |          |')
    print('      Positive |{0:6}    |{1:6}    |'.format(tp, fn))
    print('               |          |          |')
    print('Actual         |----------|----------|')
    print('               |          |          |')
    print('      Negative |{0:6}    |{1:6}    |'.format(fp, tn))
    print('               |          |          |')
    print('               |----------|----------|')


def questionA():
    # Load the training data
    M = np.genfromtxt('./mushroom.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    confusion_matrix(bagging(Xtrn, ytrn, 3, 10), Xtst, ytst)
    confusion_matrix(bagging(Xtrn, ytrn, 3, 20), Xtst, ytst)
    confusion_matrix(bagging(Xtrn, ytrn, 5, 10), Xtst, ytst)
    confusion_matrix(bagging(Xtrn, ytrn, 5, 20), Xtst, ytst)


def questionB():
    # Load the training data
    M = np.genfromtxt('./mushroom.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]
    confusion_matrix(boosting(Xtrn, ytrn, 1, 20), Xtst, ytst)
    confusion_matrix(boosting(Xtrn, ytrn, 1, 40), Xtst, ytst)
    confusion_matrix(boosting(Xtrn, ytrn, 2, 20), Xtst, ytst)
    confusion_matrix(boosting(Xtrn, ytrn, 10, 40), Xtst, ytst)


def scikit_bagging(x, y, Xtst, max_depth, num_trees):
    bg = BaggingClassifier(DecisionTreeClassifier(
        max_depth=max_depth), n_estimators=num_trees)
    bg.fit(x, y)
    y_pred = bg.predict(Xtst)
    return y_pred


def scikit_boosting(x, y, Xtst, max_depth, num_stumps):
    ada = AdaBoostClassifier(DecisionTreeClassifier(max_depth=max_depth),
                             n_estimators=num_stumps)
    ada.fit(x, y)
    y_pred = ada.predict(Xtst)
    return y_pred


def questionC():
    # Load the training data
    M = np.genfromtxt('./mushroom.train', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytrn = M[:, 0]
    Xtrn = M[:, 1:]

    # Load the test data
    M = np.genfromtxt('./mushroom.test', missing_values=0,
                      skip_header=0, delimiter=',', dtype=int)
    ytst = M[:, 0]
    Xtst = M[:, 1:]

    confusion_matrix(scikit_bagging(Xtrn, ytrn, Xtst, 3, 10), Xtst, ytst, True)
    confusion_matrix(scikit_bagging(Xtrn, ytrn, Xtst, 3, 20), Xtst, ytst, True)
    confusion_matrix(scikit_bagging(Xtrn, ytrn, Xtst, 5, 10), Xtst, ytst, True)
    confusion_matrix(scikit_bagging(Xtrn, ytrn, Xtst, 5, 20), Xtst, ytst, True)

    confusion_matrix(scikit_boosting(
        Xtrn, ytrn, Xtst, 1, 20), Xtst, ytst, True)
    confusion_matrix(scikit_boosting(
        Xtrn, ytrn, Xtst, 1, 40), Xtst, ytst, True)
    confusion_matrix(scikit_boosting(
        Xtrn, ytrn, Xtst, 2, 20), Xtst, ytst, True)
    confusion_matrix(scikit_boosting(
        Xtrn, ytrn, Xtst, 2, 40), Xtst, ytst, True)


if __name__ == '__main__':
    questionA()
    questionB()
    questionC()
    pass
