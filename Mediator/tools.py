import os
import numpy as np
import pickle


def get_information_of_data(scenario, dataset):

    root = os.path.normpath(os.getcwd() + os.sep + os.pardir)  #  os.getcwd()
    scenario_path = os.path.join("Scenario", "Scenario {}".format(scenario))
    dataset_path = os.path.join(scenario_path, "Dataset")
    dataset_path = os.path.join(dataset_path, dataset)
    src_path = os.path.join(root,dataset_path)

    test_set_path = os.path.join(src_path, 'test_set.csv')
    test_set = np.genfromtxt(test_set_path, delimiter=',')


    with open('{}/attribute_information.pkl'.format(src_path,), 'rb') as f:
        attribute_information = pickle.load(f)

    with open('{}/attributes_range.pkl'.format(src_path, ), 'rb') as f:
        attributes_range = pickle.load(f)

    with open('{}/number_target_classes.pkl'.format(src_path,), 'rb') as f:
        number_target_classes = pickle.load(f)


    return test_set, attribute_information, attributes_range, number_target_classes
