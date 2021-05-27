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

attribute_percentage = np.around(np.sqrt(len(attribute_information)) / len(attribute_information), decimals=3)


min_num_parties = number_of_parties  # 1, 5, 10, 20, 40, 80
max_num_parties = number_of_parties+1
step_size = 10

party_num_list = []
f1score_list = []
accuracy_list = []
GMean_list = []
time_list = []
# to calculate for different number of participating parties
for num_parties in range(min_num_parties, max_num_parties, step_size):

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

    included_parties_indices = np.array(range(0,num_parties))
    parties = parties_all[0:num_parties]

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

    F1_result = Prediction_and_Classification_Performance.ensemble_f1_score_for_a_set(list_of_trees, test_set)
    Accuracy_result = Prediction_and_Classification_Performance.ensemble_accuracy_for_a_set(list_of_trees, test_set)
    GMean_result = Prediction_and_Classification_Performance.ensemble_GMean_for_a_set(list_of_trees, test_set)
    print("Classification performance of the learned model, (F1 score/F1 score macro):", F1_result, "Accuracy:",
          Accuracy_result, "GMean:", GMean_result)

    party_num_list.append(num_parties)
    f1score_list.append(F1_result)
    accuracy_list.append(Accuracy_result)
    GMean_list.append(GMean_result)
    time_list.append(stop-start)


result_matrix = np.zeros((5,len(party_num_list)))
result_matrix[0, :] = np.array(party_num_list)
result_matrix[1, :] = np.array(f1score_list)
result_matrix[2, :] = np.array(accuracy_list)
result_matrix[3, :] = np.array(GMean_list)
result_matrix[4, :] = np.array(time_list)

with open('ResultMatrix.npy', 'wb') as outfile:
    np.save(outfile, result_matrix)
with open('Run_Information.txt', "w+") as outfile2:
    outfile2.write("========================================== \r\n")
    outfile2.write("Date and Time: " + (datetime.now().strftime("%d/%m/%Y %H:%M:%S"))+"\r\n")
    outfile2.write("========================================== \r\n")
    outfile2.write("Data set Information: \r\n")
    outfile2.write("Data set: " + Data_Set+"\r\n")
    outfile2.write("Number of samples: " + str(len(train_set[:, 0]) + len(test_set[:, 0])) + "\r\n")
    outfile2.write("number of attributes: %d\r\n" % len(attribute_information))
    outfile2.write("number_target_classes: %d\r\n" % number_target_classes)
    outfile2.write("attribute_information: " + str(attribute_information)+"\r\n")
    outfile2.write("attributes_range: " + str(attributes_range)+"\r\n")
    outfile2.write("Test set Proportion to Data Size: " + str(len(test_set[:, 0])/(len(train_set[:,0]) + len(test_set[:,0])))+"\r\n")
    outfile2.write("========================================== \r\n")
    outfile2.write("Run Parameters: \r\n")
    outfile2.write("Initial_seed: %d\r\n" % Initial_seed)
    outfile2.write("Number of data holder parties: %d\r\n" % number_of_parties)
    outfile2.write("Data_split_train_test_seed: %d\r\n" % Data_split_train_test_seed)
    outfile2.write("global_seed: %d\r\n" % global_seed)
    outfile2.write("attribute_percentage: " + str(attribute_percentage) + "\r\n")
    outfile2.write("number_of_trees: %d\r\n" % number_of_trees)
    outfile2.write("min_num_parties: %d\r\n" % min_num_parties)
    outfile2.write("step_size: %d\r\n" % step_size)
    outfile2.write("Secure_Aggregation_SMC: %d\r\n" % Secure_Aggregation_SMC)
    outfile2.write("Secure_Aggregation_Parameter_k: %d\r\n" % Secure_Aggregation_Parameter_k)
    outfile2.write("proportion_of_collaborating_parties: " + str(proportion_of_collaborating_parties) + "\r\n")
    outfile2.write("========================================== \r\n")
    outfile2.write("Run Results: \r\n")
    outfile2.write("party_num: " + np.array2string(result_matrix[0, :])+"\r\n")
    outfile2.write("F1 Score: " + np.array2string(result_matrix[1, :])+"\r\n")
    outfile2.write("Accuracy: " + np.array2string(result_matrix[2, :])+"\r\n")
    outfile2.write("GMean: " + np.array2string(result_matrix[3, :])+"\r\n")
    outfile2.write("Learning Time: " + np.array2string(result_matrix[4, :])+"\r\n")
    outfile2.write("========================================== \r\n")
print("Results saved!")
