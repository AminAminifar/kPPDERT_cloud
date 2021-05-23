import os
import numpy as np


def _find_range_of_data(data, attribute_information):
    attributes_range = []
    for i in range(len(attribute_information)):
        if attribute_information[i] == ["categorical"]:
            unique = np.unique(data[:, i], return_counts=False)
            attributes_range.append(unique)
        else:
            range_min = np.amin(data[:, i])
            range_max = np.amax(data[:, i])
            attributes_range.append([range_min, range_max])
    return attributes_range


def get_chunk_of_data(username, dataset):
    root = os.getcwd()
    src_path = root + '/DistributedDatasets/' + dataset
    train_set = np.genfromtxt(src_path + '/tr_party_num_' + username + '.csv', delimiter=',')
    test_set = np.genfromtxt(src_path + '/ts_party_num_' + username + '.csv', delimiter=',')

    # train_set = train_set[0:100, :]

    attribute_information = [["continuous"],
                             ["categorical"],
                             ["continuous"],
                             ["categorical"],
                             ["continuous"],
                             ["categorical"],
                             ["categorical"],
                             ["categorical"],
                             ["categorical"],
                             ["categorical"],
                             ["continuous"],
                             ["continuous"],
                             ["continuous"],
                             ["categorical"]]

    train_indices = []
    test_indices = []
    all_train_indices = []

    src_path = root + '/src/Datasets/' + dataset
    all_train_set = np.genfromtxt(src_path + '/Train_Data' + '.csv', delimiter=',')

    unique = np.unique(all_train_set[:, -1], return_counts=False)

    for i in range(len(unique)):
        train_indices.append(train_set[:, -1] == unique[i])
        all_train_indices.append(all_train_set[:, -1] == unique[i])
        test_indices.append(test_set[:, -1] == unique[i])
    for i in range(len(unique)):
        train_set[train_indices[i], -1] = int(i)
        all_train_set[all_train_indices[i], -1] = int(i)
        test_set[test_indices[i], -1] = int(i)

    number_target_classes = len(unique)

    attributes_range = _find_range_of_data(all_train_set, attribute_information)

    return train_set, test_set, attribute_information, attributes_range, number_target_classes


def get_information_of_data(dataset):

    attribute_information = [["continuous"],
                             ["categorical"],
                             ["continuous"],
                             ["categorical"],
                             ["continuous"],
                             ["categorical"],
                             ["categorical"],
                             ["categorical"],
                             ["categorical"],
                             ["categorical"],
                             ["continuous"],
                             ["continuous"],
                             ["continuous"],
                             ["categorical"]]

    train_indices = []
    test_indices = []
    root = os.getcwd()
    src_path = root + '/src/Datasets/' + dataset
    train_set = np.genfromtxt(src_path + '/Train_Data' + '.csv', delimiter=',')
    test_set = np.genfromtxt(src_path + '/Test_Data' + '.csv', delimiter=',')

    unique = np.unique(train_set[:, -1], return_counts=False)

    for i in range(len(unique)):
        train_indices.append(train_set[:, -1] == unique[i])
        test_indices.append(test_set[:, -1] == unique[i])
    for i in range(len(unique)):
        train_set[train_indices[i], -1] = int(i)
        test_set[test_indices[i], -1] = int(i)

    number_target_classes = len(unique)

    attributes_range = _find_range_of_data(train_set, attribute_information)

    return test_set, attribute_information, attributes_range, number_target_classes
