import generate_parties
import Server_Parties_Interface
import server_class
import Prediction_and_Classification_Performance
import Import_Data
import numpy as np
import random
import timeit
from datetime import datetime
import model_selection
from sklearn.metrics import f1_score, accuracy_score, confusion_matrix, matthews_corrcoef

Initial_seed = random.randint(1, 10 ** 5)
random.seed(Initial_seed)

# IMPORT DATA
# options (data sets):{
# mfeat (Multiple Features)
# Nursery
# Adult
# Waveform
# }
Data_Set = "Waveform"
train_set, test_set, attribute_information, number_target_classes = Import_Data.import_data(Data_Set)
attributes_range = Import_Data.find_range_of_data(train_set, attribute_information)

# Settings
Data_split_train_test_seed = random.randint(1, 10 ** 5)
global_seed = random.randint(1, 10 ** 5)
seed_common = random.randint(1, 10 ** 5)
number_of_parties = 1  # 80
number_of_trees = 25

# What proportion of parties participate in selecting the best candidate decision node/leaf:
proportion_of_collaborating_parties = .7  # can be changed to a value<1
num_participating_parties = round(number_of_parties*proportion_of_collaborating_parties)

Secure_Aggregation_SMC = True  # False # True if simulation of SMC part is required
# min number of colluding (data holder) parties (must be lees than num_participating_parties):
Secure_Aggregation_Parameter_k = num_participating_parties-1  # can be changed to a value<num_participating_parties


# MODEL SELECTION:
# select the best attribute percentage by 5-fold model selection,
# and with respect to classification performance (F1-Score).
# best_val = model_selection.select_attribute_percentage(global_seed=global_seed, number_of_parties=1,
#                                                        number_of_trees=number_of_trees,
#                                                        train_set=train_set,
#                                                        attribute_information=attribute_information,
#                                                        number_target_classes=number_target_classes,
#                                                        attributes_range=attributes_range)

# attribute_percentage = best_val

# Or set manually. We already conducted model selection,
# the following are the results:
# Adult: .9, Waveform: .2,Nursery: .9, Multiple features: .1
# default value:
attribute_percentage = np.around(np.sqrt(len(attribute_information)) / len(attribute_information), decimals=3)




groundTruth = []
prediction_list_kPPDERT = []

parties_all = generate_parties.generate(global_seed=global_seed, number_of_parties=number_of_parties,
                                        train_set=train_set,
                                        attribute_information=attribute_information,
                                        number_target_classes=number_target_classes,
                                        attributes_range=attributes_range,
                                        attribute_percentage=attribute_percentage,
                                        Secure_Aggregation_SMC=Secure_Aggregation_SMC,
                                        Secure_Aggregation_Parameter_k=Secure_Aggregation_Parameter_k,
                                        seed_common=seed_common,
                                        num_participating_parties=num_participating_parties,
                                        Data_split_train_test_seed=Data_split_train_test_seed)

included_parties_indices = np.array(range(0,number_of_parties))
parties = parties_all[0:number_of_parties]

# instantiate Interface class for communications between server and parties
Interface = Server_Parties_Interface.interface(parties, proportion_of_collaborating_parties, number_of_parties)

# initialization
if Secure_Aggregation_SMC:
    Interface.initialize_parties()
    print("Initialization Done!")

# instantiate server
server = server_class.server(global_seed=global_seed, attribute_range=attributes_range,
                             attribute_info=attribute_information,
                             num_target_classes=number_target_classes,
                             aggregator_func=Interface.aggregator,
                             parties_update_func=Interface.parties_update,
                             attribute_percentage=attribute_percentage,
                             included_parties_indices=included_parties_indices,
                             Secure_Aggregation_SMC=Secure_Aggregation_SMC,
                             parties_reset_func=Interface.parties_reset)

print("========================================")
print("LEARNING...")
start = timeit.default_timer()
list_of_trees = server.make_tree_group(impurity_measure='entropy', num_of_trees=number_of_trees)
stop = timeit.default_timer()
print("A Group of ", number_of_trees, "Trees are Learned!")
print("Elapsed Time: ", stop-start, " Sec")
print("========================================")
print("CLASSIFICATION PERFORMANCE...")

prediction, true_labels = \
    Prediction_and_Classification_Performance.get_result_vectors(list_of_trees, test_set)

groundTruth.append(true_labels)
prediction_list_kPPDERT.append(prediction)


def print_results(labels_vec, predictions_vec):
    # tn, fp, fn, tp = confusion_matrix(labels_vec, predictions_vec).ravel()
    f1_performance = f1_score(labels_vec, predictions_vec, average='weighted')
    acc_performance = accuracy_score(labels_vec, predictions_vec)
    mcc_performance = matthews_corrcoef(labels_vec, predictions_vec)
    # print("tn, fp, fn, tp: ", tn, fp, fn, tp)
    print("f1_performance: ", f1_performance)
    print("acc_performance", acc_performance)
    print("mcc_performance: ", mcc_performance)


print("CLASSIFICATION PERFORMANCE...")
groundTruth_vec = np.concatenate(np.asarray(groundTruth))
prediction_kPPDERT_vec = np.concatenate(np.asarray(prediction_list_kPPDERT))
print("groundTruth_vec and prediction_ert_vec:")
print_results(groundTruth_vec, prediction_kPPDERT_vec)

