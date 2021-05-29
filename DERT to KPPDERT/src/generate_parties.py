from src import party_class
import numpy as np


def generate(global_seed, train_set, attribute_information, number_target_classes, attributes_range):

    party = party_class.party(global_seed=global_seed,
                              data_subset=train_set,
                              attribute_range=attributes_range,
                              attribute_info=attribute_information,
                              num_target_classes=number_target_classes)

    print("a party have been created.\n\
    Data sample randomly assigned to this party.\n\
    The proportion of number of samples for each party is almost equal.")
    return party
