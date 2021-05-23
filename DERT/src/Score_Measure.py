import numpy as np

#_Score Measure
def Gini_index(input_vec): ## This may be checked later for error in the result
    counts = input_vec
    gini_index_val = 1
    for i in range(counts.shape[0]):
        gini_index_val -= (counts[i]/sum(counts))**2
    return gini_index_val

from scipy.stats import entropy
def Entropy(input_vec):
    counts = input_vec
    entropy_val = entropy(counts, base=2)
    return entropy_val


def purity(true_set_classes, false_set_classes, impurity_measure='entropy'):
    """Checks the purity of sets passed to it based on their labels
    information gain of the criterion used to divide the dataset
    input: classes for the true and false sets along with
    the specification of impurity measure
    output: the gain or purity of such division"""

    All_classes = np.array(true_set_classes + false_set_classes)

    if impurity_measure == 'gini':
        purity = Gini_index(All_classes) - \
                 (sum(true_set_classes) / sum(All_classes)) * Gini_index(true_set_classes) - \
                 (sum(false_set_classes) / sum(All_classes)) * Gini_index(false_set_classes)
    else:
        # if purity measure is information gain
        purity = Entropy(All_classes) - \
                 (sum(true_set_classes) / sum(All_classes)) * Entropy(true_set_classes) - \
                 (sum(false_set_classes) / sum(All_classes)) * Entropy(false_set_classes)

    return purity
