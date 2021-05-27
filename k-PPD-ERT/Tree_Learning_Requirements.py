import Score_Measure
import numpy as np
import random

class Tree_Learning_Requirements:

    def __init__(self, global_seed, attribute_range, attribute_info, num_target_classes,\
                 aggregator_func, parties_update_func, attribute_percentage,\
                 included_parties_indices, Secure_Aggregation_SMC):
        self.attribute_range = attribute_range
        self.attribute_info = attribute_info
        self.num_target_classes = num_target_classes
        self.aggregator_func = aggregator_func
        self.parties_update_func = parties_update_func
        self.global_seed = global_seed
        self.global_random_state = None
        random.seed(global_seed)
        self.num_transactions = 0
        self.num_updates = 0
        self.att_perc = attribute_percentage
        self.included_parties_indices = included_parties_indices
        self.Secure_Aggregation_SMC = Secure_Aggregation_SMC

    #_Tree Learning Requirements
    class Criterion:
        """The criterion objects instantiated from Criterion class are used to divide a dataset into two sets"""

        def __init__(self, attribute_type, attribute_index, point_or_category):
            self.attribute_type = attribute_type
            self.attribute_index = attribute_index
            self.point_or_category = point_or_category

        def check(self, sampel):
            """for numerical attributes: Check if the value in the specified attribute of the sampel is
            greater or less than the specified point.
            for categorical attributes: check if the sample' attribute is in the same as specified category"""

            if self.attribute_type == ["categorical"]:
                result = sampel[self.attribute_index] == self.point_or_category
            else:
                result = sampel[self.attribute_index] >= self.point_or_category
            return result


    def Pick_a_random_split(self, attribute_indices):
        """SELECT SPLITS RANDOMELY FOR THE ATTRIBUTES
        input: range of attributes (fore categoricals: categories); info about attribute(categrical vs numerical);
        indices of several attributes selected ranomly;
        output: a list containing the randome splits for those randomly selected attributes."""

        split_list = []
        for i in range(len(attribute_indices)):
            index_of_the_attribute = attribute_indices[i]
            attribute_type = self.attribute_info[index_of_the_attribute]
            if attribute_type == ["categorical"]:
                #  find all the available categories in the attribute
                unique = self.attribute_range[index_of_the_attribute]
                #  selectr one of the categories randomly
                rnd_split = random.choice(unique)
                #  append the generated split to the list of splits for all randomely selected attributes
                split_list.append(rnd_split)
            else:
                #  min and max in the range of the attribute
                range_min = self.attribute_range[index_of_the_attribute][0]
                range_max = self.attribute_range[index_of_the_attribute][1]
                #  generate a randome number in the range of the attribute
                rnd_split = random.uniform(range_min, range_max)
                #  append the generated split to the list of splits for all randomely selected attributes
                split_list.append(rnd_split)

        return split_list


    def find_best_attribute(self, impurity_measure='entropy', node_id=0, branch=None):
        """Find the best attribute for using in the node for classification.
        input: a subset of data, including all the attributes and the labels.
        output: best_value (best purity value),...
        best_criterion (best attribute to be used for classification along with the criterion)."""

        def stop_split(classses_vec, n_min):
            flag = False
            if sum(classses_vec) < n_min:
                flag = True
            else:
                classses_vec = np.array(classses_vec)
                if np.count_nonzero(classses_vec) < 2:
                    flag = True
            return flag

        best_value = 0  # best value for purity measure (with information gain or gini index)
        best_criterion = None  # best attribute along with the criterion for classification
        classes_for_samples = []  # classes for samples meeting the criteria

        ##CHOOSE A VALUE FOR "n_min" IN THE ERT ALGORITHM
        n_min = 5

        ##SAVE THE STATE OF THE RANDOM FUNCTION
        random_func_state = random.getstate()

        ##SELECT ATTRIBUTES RANDOMELY
        #  number of attributes
        num_att = len(self.attribute_info)
        #  percentage of attributes randomely selected
        att_perc = self.att_perc#.7 #.5
        #  number of attributes randomely selected
        num_rnd_att = np.int(round(att_perc * num_att))
        # indeces of randomly selected attributes
        rand_attribute_index = [random.randint(0, num_att - 1) for p in range(0, num_rnd_att)]

        ##SELECT RANDOM SPLITS
        split_list = self.Pick_a_random_split(rand_attribute_index)

        ##CHOOSE BEST {ATTRIBUTE,SPLIT} SET
        criterion_list = []
        for i in range(num_rnd_att):  # for each selected attribute
            index_of_the_attribute = rand_attribute_index[i]
            attribute_type = self.attribute_info[index_of_the_attribute]
            rnd_split = split_list[i]
            #  make a citerion to evaluate
            criterion_list.append(self.Criterion(attribute_type, index_of_the_attribute, rnd_split))

        # divide the data based on the selected attribute and split
        true_set_clasess, false_set_clasess = self.aggregator_func(node_id, branch, random_func_state,
                                                         self.num_target_classes, num_criteria=num_rnd_att)
        self.num_transactions +=1

        ## FOR SMC PART
        # if self.Secure_Aggregation_SMC:
        #     pass



        for i in range(len(criterion_list)):
            stop_flag = stop_split(((true_set_clasess[i, :]) + (false_set_clasess[i, :])), n_min)  ### change...
            if not (sum(true_set_clasess[i, :]) == 0 or sum(
                    false_set_clasess[i, :]) == 0 or stop_flag):  ## to be checked in case of error
                # calculate the purity measure after division
                # adopting the above criterion
                purity_val = Score_Measure.purity(true_set_clasess[i, :], false_set_clasess[i, :], impurity_measure)
                if purity_val > best_value:
                    best_value, best_criterion = purity_val, criterion_list[i]

        if best_value > 0:
            self.parties_update_func(best_criterion, node_id, branch)
            node_id += 1
            self.num_updates += 1
        else:
            classes_for_samples = true_set_clasess[0, :] + false_set_clasess[0, :]

        return best_value, best_criterion, classes_for_samples, node_id
