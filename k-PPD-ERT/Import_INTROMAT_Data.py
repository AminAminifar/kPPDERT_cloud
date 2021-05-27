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
    if dataset_name == "depression_new_approach":
        data_set = np.genfromtxt('Datasets/depression/data_all_with_patient_index.csv',
                                 delimiter=',')  # data_all_with_patient_index_100_60by24
        attribute_information = []
        for i in range(60 * 24):
            attribute_information.append(["continuous"])

    unique = np.unique(data_set[:, -1], return_counts=False)
    number_target_classes = len(unique)

    return data_set, attribute_information, number_target_classes