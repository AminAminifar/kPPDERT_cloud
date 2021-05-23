import party_class
import numpy as np
import random


def generate(global_seed, number_of_parties, train_set, attribute_information,
             number_target_classes, attributes_range,
             attribute_percentage, Secure_Aggregation_SMC, seed_common,
             Secure_Aggregation_Parameter_k, num_participating_parties,
             Data_split_train_test_seed=None):
    number_of_training_samples = train_set.shape[0]
    # Shuffle data samples
    np.random.seed(seed=Data_split_train_test_seed)
    indices = np.arange(number_of_training_samples)
    np.random.shuffle(indices)

    data_subsets_indices = np.array_split(indices, number_of_parties)

    parties = []
    for i in range(number_of_parties):
        parties.append(party_class.Party(global_seed=global_seed,
                                         data_subset=train_set[data_subsets_indices[i],:],
                                         attribute_range=attributes_range,
                                         attribute_info=attribute_information,
                                         num_target_classes=number_target_classes,
                                         attribute_percentage=attribute_percentage,
                                         spp=seed_common,
                                         num_participating_parties=num_participating_parties,
                                         secure_aggregation_smc=Secure_Aggregation_SMC,
                                         secure_aggregation_parameter_k=Secure_Aggregation_Parameter_k,
                                         num_parties=number_of_parties,
                                         party_id=i))

    print("A list of parties have been created.\n\
    Data samples radomely assigned to each party.\n\
    The proportion of number of samples for each party is almost equal.")

    return parties
