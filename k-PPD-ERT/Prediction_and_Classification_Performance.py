import Tree_Elements
import numpy as np
from sklearn.metrics import f1_score,accuracy_score,confusion_matrix

#_Prediction and Classification Performance
def Tree_Predict(tree, sampel):
    if isinstance(tree, Tree_Elements.End_Node):
        prediction = tree.label
    else:
        if tree.criterion.check(sampel):
            prediction = Tree_Predict(tree.true_branch, sampel)
        else:
            prediction = Tree_Predict(tree.false_branch, sampel)
    return prediction


def ensemble_get_vectors(tree_group, data):
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

    prediction = np.array(prediction)
    prediction = prediction.astype(int)
    # prediction = prediction.reshape(-1,1);print(prediction.shape)
    prediction = prediction + 1
    true_labels = np.array(true_labels)
    true_labels = true_labels.astype(int)
    # true_labels = true_labels.reshape(-1,1);print(true_labels.shape)
    true_labels = true_labels + 1

    unique = np.unique(true_labels, return_counts=False)
    return prediction, true_labels, unique

def ensemble_f1_score_for_a_set(tree_group, data):

    prediction, true_labels, unique = ensemble_get_vectors(tree_group, data)
    # print("prediction",len(prediction),"true_labels",len(true_labels))
    # print("prediction",(prediction),"true_labels",(true_labels))

    if len(unique) > 2:  # multi class
        performance = f1_score(true_labels, prediction, labels=unique, average='macro')
    else:  # binary class
        performance = f1_score(true_labels, prediction, labels=unique, average='weighted')
        if performance==0:
            tn, fp, fn, tp = confusion_matrix(true_labels, prediction).ravel()
            # print("tn, fp, fn, tp",tn, fp, fn, tp)
            if tp!=0:
                performance = tp/(tp + .5*(fp+fn))
            else:
                performance = tn/(tn + .5*(fp + fn))




    return performance

def ensemble_accuracy_for_a_set(tree_group, data):

    prediction, true_labels, unique = ensemble_get_vectors(tree_group, data)

    performance = accuracy_score(true_labels, prediction)

    return performance

def ensemble_GMean_for_a_set(tree_group, data):

    prediction, true_labels, unique = ensemble_get_vectors(tree_group, data)

    for i in range(len(prediction)):
        if prediction[i]!=true_labels[i]:
            prediction[i] = 10^5#[None]

    prediction_unique, prediction_count = np.unique(prediction, return_counts=True)
    true_labels_unique, true_labels_count = np.unique(true_labels, return_counts=True)
    true_to_all_ratio = np.zeros(len(unique))
    for label in unique:
        if label in prediction_unique:
            index = np.where(prediction_unique==label)[0][0]
            true_num = prediction_count[index]
        else:
            true_num = 0
        index = np.where(true_labels_unique == label)[0][0]
        all_num = true_labels_count[index]

        true_to_all_ratio[index] = true_num/all_num

    n = len(unique)
    product = np.prod(true_to_all_ratio)
    if product>0:
        performance = (product)**(1.0/n)
    else:
        performance = product

    return performance

def get_result_vectors(tree_group, data):
    prediction, true_labels, _ = ensemble_get_vectors(tree_group, data)
    return prediction, true_labels