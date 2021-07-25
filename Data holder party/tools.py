import os
import numpy as np
import pickle
import pandas as pd


def get_chunk_of_data(username, dataset, scenario):
    root = os.path.normpath(os.getcwd() + os.sep + os.pardir)  #  os.getcwd()
    scenario_path = os.path.join("Scenario", "Scenario {}".format(scenario))
    dataset_path = os.path.join(scenario_path, "Dataset")
    dataset_path = os.path.join(dataset_path, dataset)
    src_path = os.path.join(root,dataset_path)

    train_set_path = os.path.join(src_path, 'tr_party_num_' + username + '.csv')
    train_set = pd.read_csv(train_set_path, sep=',').values

    attribute_information_path = os.path.join(src_path, 'attribute_information.pkl')
    with open(attribute_information_path, 'rb') as f:
        attribute_information = pickle.load(f)

    attributes_range_path = os.path.join(src_path, 'attributes_range.pkl')
    with open(attributes_range_path, 'rb') as f:
        attributes_range = pickle.load(f)

    number_target_classes_path = os.path.join(src_path, 'number_target_classes.pkl')
    with open(number_target_classes_path, 'rb') as f:
        number_target_classes = pickle.load(f)

    return train_set, attribute_information, attributes_range, number_target_classes

