import os
import numpy as np
import pickle


def get_chunk_of_data(username, dataset, scenario):
    root = os.getcwd()
    src_path = "{}\\Scenario\\Scenario {}\\Dataset\\{}".format(root, scenario, dataset)
    train_set = np.genfromtxt(src_path + '\\tr_party_num_' + username + '.csv', delimiter=',')

    with open('{}/attribute_information.pkl'.format(src_path,), 'rb') as f:
        attribute_information = pickle.load(f)

    with open('{}/attributes_range.pkl'.format(src_path, ), 'rb') as f:
        attributes_range = pickle.load(f)

    with open('{}/number_target_classes.pkl'.format(src_path,), 'rb') as f:
        number_target_classes = pickle.load(f)

    train_indices = []
    all_train_indices = []

    all_train_set = np.genfromtxt(src_path + '/train_set' + '.csv', delimiter=',')

    unique = np.unique(all_train_set[:, -1], return_counts=False)

    for i in range(len(unique)):
        train_indices.append(train_set[:, -1] == unique[i])
        all_train_indices.append(all_train_set[:, -1] == unique[i])
    for i in range(len(unique)):
        train_set[train_indices[i], -1] = int(i)
        all_train_set[all_train_indices[i], -1] = int(i)

    return train_set, attribute_information, attributes_range, number_target_classes

