from sklearn.model_selection import train_test_split
import numpy as np


def find_range_of_data(data, attribute_information):
    attributes_range = []
    for i in range(len(attribute_information)):
        if attribute_information[i]==["categorical"]:
            unique = np.unique(data[:,i], return_counts=False)
            attributes_range.append(unique)
        else:
            range_min = np.amin(data[:,i])
            range_max = np.amax(data[:,i])
            attributes_range.append([range_min, range_max])
    return attributes_range



def import_data(dataset_name="Adult"):
    if dataset_name == "Adult":
        train_set = np.genfromtxt('Datasets\Adult\Train_Data.csv', delimiter=',')
        test_set = np.genfromtxt('Datasets\Adult\Test_Data.csv', delimiter=',')
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
    elif dataset_name == "Waveform":
        # import
        data_set = np.genfromtxt('Datasets\Waveform\waveform.csv', delimiter=',')
        # split data to train and test sets
        train_set, test_set = train_test_split(data_set, test_size=1/3, random_state=0)
        attribute_information = [["continuous"], ["continuous"], ["continuous"], ["continuous"],
                                 ["continuous"], ["continuous"], ["continuous"], ["continuous"],
                                 ["continuous"], ["continuous"], ["continuous"], ["continuous"],
                                 ["continuous"], ["continuous"], ["continuous"], ["continuous"],
                                 ["continuous"], ["continuous"], ["continuous"], ["continuous"], ["continuous"]]
    elif dataset_name == "Nursery":
        # import
        data_set = np.genfromtxt('Datasets/Nursery/nursery.csv', delimiter=',')
        # split data to train and test sets
        train_set, test_set = train_test_split(data_set, test_size=1/3, random_state=0)
        attribute_information = [["categorical"],
                                 ["categorical"],
                                 ["categorical"],
                                 ["categorical"],
                                 ["categorical"],
                                 ["categorical"],
                                 ["categorical"],
                                 ["categorical"]]
    elif dataset_name == "mfeat":
        # import
        data_set = np.genfromtxt('Datasets/mfeat/mfeat.csv', delimiter=',')
        # split data to train and test sets
        train_set, test_set = train_test_split(data_set, test_size=1/3, random_state=0)
        attribute_information = []
        for i in range(649):
            attribute_information.append(["continuous"])

    # data preparation
    # if the labels do not start from 0; this makes it start from 0 and goes up
    train_indices = []
    test_indices = []
    unique = np.unique(train_set[:, -1], return_counts=False)
    for i in range(len(unique)):
        train_indices.append(train_set[:, -1] == unique[i])
        test_indices.append(test_set[:, -1] == unique[i])
    for i in range(len(unique)):
        train_set[train_indices[i], -1] = int(i)
        test_set[test_indices[i], -1] = int(i)

    unique = np.unique(train_set[:, -1], return_counts=False)
    number_target_classes = len(unique)

    return train_set, test_set, attribute_information, number_target_classes
