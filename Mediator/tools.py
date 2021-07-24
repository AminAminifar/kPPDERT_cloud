import os
import numpy as np
import pickle


def get_information_of_data(scenario, dataset):

    root = os.path.normpath(os.getcwd() + os.sep + os.pardir)  #  os.getcwd()
    src_path = "{}\\Scenario\\Scenario {}\\Dataset\\{}".format(root, scenario, dataset)

    train_indices = []
    test_indices = []

    train_set = np.genfromtxt(src_path + '\\train_set' + '.csv', delimiter=',')
    test_set = np.genfromtxt(src_path + '\\test_set' + '.csv', delimiter=',')

    with open('{}/attribute_information.pkl'.format(src_path,), 'rb') as f:
        attribute_information = pickle.load(f)

    with open('{}/attributes_range.pkl'.format(src_path, ), 'rb') as f:
        attributes_range = pickle.load(f)

    with open('{}/number_target_classes.pkl'.format(src_path,), 'rb') as f:
        number_target_classes = pickle.load(f)

    unique = np.unique(train_set[:, -1], return_counts=False)

    for i in range(len(unique)):
        train_indices.append(train_set[:, -1] == unique[i])
        test_indices.append(test_set[:, -1] == unique[i])
    for i in range(len(unique)):
        train_set[train_indices[i], -1] = int(i)
        test_set[test_indices[i], -1] = int(i)

    return test_set, attribute_information, attributes_range, number_target_classes
