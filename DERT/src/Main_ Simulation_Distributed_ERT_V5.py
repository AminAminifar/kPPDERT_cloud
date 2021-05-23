import generate_parties
import Server_Parties_Interface
import server_class
import Prediction_and_Classification_Performance

import Import_Data
import numpy as np
import random

random.seed(13)

##IMPORT DATA
# options (data sets):{
# Adult
# Cryotherapy
# Datasets Healthy Older People:set1
# Datasets Healthy Older People:set2
# Diabetic Retinopathy Debrecen
# Drug consumption (quantified):Alcohol
# Waveform
# Waveform+noise
# }
####################TTTTTTTTTTTTTTTTOOOOOOOOOOBBBBBBBBEEEEEEchanged########################################
# train_set, test_set, attribute_information, number_target_classes = Import_Data.import_data("Nursery")
train_set_temp, test_set_temp, attribute_information, number_target_classes = Import_Data.import_data("Nursery")
train_set, test_set = train_set_temp[0:100,:] , test_set_temp[0:100,:]
####################TTTTTTTTTTTTTTTTOOOOOOOOOOBBBBBBBBEEEEEEchanged########################################
attributes_range = Import_Data.find_range_of_data(train_set, attribute_information)



#generate parties: it instantiates several objects from party class
parties = generate_parties.generate(global_seed=13, number_of_parties=2, train_set=train_set,
                                    attribute_information=attribute_information,
                                    number_target_classes=number_target_classes,
                                    attributes_range=attributes_range)

#instantiate Interface class for communications between server and parties
Interface = Server_Parties_Interface.interface(parties)

#instantiate server
server = server_class.server(global_seed=13, attribute_range=attributes_range,
                             attribute_info=attribute_information,
                             num_target_classes=number_target_classes,
                             aggregator_func=Interface.aggregator,
                             parties_update_func=Interface.parties_update)


print("========================================")
print("LEARNING...")
#LEARNING
f_tree = server.grow_tree(impurity_measure='gini')
print("f_tree is Learned!")
list_of_trees = server.make_tree_group(impurity_measure='entropy', num_of_trees=10)
print("A Group of ", "10", "Trees are Learned!")



print("========================================")
print("CLASSIFICATION PERFORMANCE...")
#PRINT CLASSIFICATION PERFORMANCE
print("Classification performance for only one tree (F1 score/F1 score macro):",\
      Prediction_and_Classification_Performance.f1_score_for_a_set(f_tree, test_set))
print("Classification performance for several trees (F1 score/F1 score macro):",\
      Prediction_and_Classification_Performance.ensemble_f1_score_for_a_set(list_of_trees, test_set))


print("========================================")
print("Comparison to SKLearn Library Classifier...")
#_Comparison to SKLearn Library Classifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score
clf = ExtraTreesClassifier(n_estimators=10, random_state=0)
clf.fit(train_set[:,0:-1], train_set[:,-1])
prediction = clf.predict(test_set[:,0:-1])
prediction = np.array(prediction); prediction = prediction + 1

unique = np.unique(test_set[:,-1], return_counts=False)
if len(unique)>2: #multi class
    performance = f1_score(test_set[:,-1] + 1, prediction, average='macro')
else:  #binary class
    performance = f1_score(test_set[:,-1] + 1, prediction)
print(performance)