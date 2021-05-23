import generate_parties
import Server_Parties_Interface
import server_class
import Prediction_and_Classification_Performance


import numpy as np
import random

from sklearn.model_selection import KFold

def select_attribute_percentage(global_seed, number_of_parties, number_of_trees,\
                                train_set, attribute_information, number_target_classes, attributes_range):
    f1_val_1 = np.zeros(9)
    i1 = 0
    possible_values = [.1, .2, .3, .4, .5, .6, .7, .8, .9]  # [.25,.5,.75]
    for val in possible_values:

        # Settings
        attribute_percentage = val
        # generate personal random seeds
        personal_random_seeds = [random.randint(0, 10000 * number_of_parties) for p in range(0, number_of_parties)]

        kf = KFold(n_splits=5, random_state=None, shuffle=False)
        f1_val_2 = np.zeros(5)
        i2 = 0
        for train_index, test_index in kf.split(train_set):
            train_set_val = train_set[train_index, :]
            test_set_val = train_set[test_index, :]
            # generate parties: it instantiates several objects from party class
            parties = generate_parties.generate(global_seed=global_seed, number_of_parties=number_of_parties,
                                                train_set=train_set_val, \
                                                attribute_information=attribute_information, \
                                                number_target_classes=number_target_classes, \
                                                attributes_range=attributes_range, \
                                                personal_random_seeds=personal_random_seeds, \
                                                attribute_percentage=attribute_percentage)

            # instantiate Interface class for communications between server and parties
            Interface = Server_Parties_Interface.interface(parties)

            # instantiate server
            server = server_class.server(global_seed=global_seed, attribute_range=attributes_range, \
                                         attribute_info=attribute_information, \
                                         num_target_classes=number_target_classes, \
                                         aggregator_func=Interface.aggregator, \
                                         parties_update_func=Interface.parties_update, \
                                         personal_random_seeds=personal_random_seeds, \
                                         attribute_percentage=attribute_percentage,
                                         included_parties_indices=np.array(range(0,number_of_parties)) )

            print("========================================")
            print("LEARNING...")
            # LEARNING
            # f_tree = server.grow_tree(impurity_measure='gini')
            # print("f_tree is Learned!")
            list_of_trees = server.make_tree_group(impurity_measure='entropy', num_of_trees=number_of_trees)
            print("A Group of ", number_of_trees, "Trees are Learned!")
            print("========================================")
            print("CLASSIFICATION PERFORMANCE...")

            F1_result = Prediction_and_Classification_Performance.ensemble_f1_score_for_a_set(list_of_trees,
                                                                                              test_set_val)
            Accuracy_result = Prediction_and_Classification_Performance.ensemble_accuracy_for_a_set(list_of_trees,
                                                                                                    test_set_val)
            print("Classification performance for several trees (F1 score/F1 score macro):", F1_result, "Accuracy:",
                  Accuracy_result)
            f1_val_2[i2] = F1_result
            i2 += 1
            print("f1_val_2: ", f1_val_2)
        f1_val_1[i1] = sum(f1_val_2) / len(f1_val_2)
        i1 += 1
        print("f1_val_1: ", f1_val_1)

    val = possible_values[np.argmax(f1_val_1)];
    print("selected val: ", val)
    return val
