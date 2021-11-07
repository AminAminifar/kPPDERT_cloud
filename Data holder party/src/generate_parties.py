from src import party_class
import numpy as np
import random


def generate(global_seed, number_of_parties, train_set, attribute_information,
             number_target_classes, attributes_range,
             attribute_percentage, Secure_Aggregation_SMC, seed_common,
             Secure_Aggregation_Parameter_k, num_participating_parties, party_id, scenario):
    party = party_class.Party(global_seed=global_seed,
                              data_subset=train_set,
                              attribute_range=attributes_range,
                              attribute_info=attribute_information,
                              num_target_classes=number_target_classes,
                              attribute_percentage=attribute_percentage,
                              spp=seed_common,
                              num_participating_parties=num_participating_parties,
                              secure_aggregation_smc=Secure_Aggregation_SMC,
                              secure_aggregation_parameter_k=Secure_Aggregation_Parameter_k,
                              num_parties=number_of_parties,
                              party_id=party_id,
                              scenario=scenario
                              )

    print("A party have been created.\n\
    Data samples randomly assigned to the party.\n\
    The proportion of samples for each party is almost equal.")

    return party
