import src.Tree_Elements
import numpy as np
from sklearn.metrics import f1_score

#_Prediction and Classification Performance
def Tree_Predict(tree, sampel):
    if isinstance(tree, src.Tree_Elements.End_Node):
        prediction = tree.label
    else:
        if tree.criterion.check(sampel):
            prediction = Tree_Predict(tree.true_branch, sampel)
        else:
            prediction = Tree_Predict(tree.false_branch, sampel)
    return prediction




def f1_score_for_a_set(tree, data):
    """calculates the f1_score of a tree for a set of data
    input: a tree and a set of data
    output: f1_score value for classification for the given data based on the given tree"""

    true_labels = data[:, data.shape[1] - 1]
    prediction_length = len(true_labels)  # to be checked in case of error
    prediction = [None] * prediction_length
    for i in range(prediction_length):
        prediction[i] = Tree_Predict(tree, data[i, :])

    prediction = np.array(prediction);
    prediction = prediction + 1
    true_labels = np.array(true_labels);
    true_labels = true_labels + 1

    unique = np.unique(true_labels, return_counts=False)
    if len(unique) > 2:  # multi class
        performance = f1_score(true_labels, prediction, average='macro')
    else:  # binary class
        performance = f1_score(true_labels, prediction)

    return performance


def ensemble_f1_score_for_a_set(tree_group, data):
    true_labels = data[:, data.shape[1] - 1]
    prediction_length = len(true_labels)  # to be checked in case of error
    prediction = [None] * prediction_length
    votes_length = len(tree_group)
    votes = [None] * votes_length

    for i in range(prediction_length):
        for j in range(votes_length):
            votes[j] = Tree_Predict(tree_group[j], data[i, :])
        unique, counts = np.unique(votes, return_counts=True)
        prediction[i] = unique[np.argmax(counts)]

    prediction = np.array(prediction);
    prediction = prediction + 1
    true_labels = np.array(true_labels);
    true_labels = true_labels + 1

    unique = np.unique(true_labels, return_counts=False)
    if len(unique) > 2:  # multi class
        performance = f1_score(true_labels, prediction, average='macro')
    else:  # binary class
        performance = f1_score(true_labels, prediction)

    return performance
