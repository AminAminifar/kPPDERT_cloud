import numpy as np
import random
import k_PPD_ERT_functions


class Party:
    """every party will have part of the data (data subset)
    every party has the function for returning the class labels (based on the criteria)
    of its samples to the aggregator"""

    def __init__(self, global_seed, data_subset, attribute_range, attribute_info, num_target_classes,
                 attribute_percentage, spp, num_participating_parties,
                 secure_aggregation_smc, secure_aggregation_parameter_k, num_parties, party_id):
        self.data_subset = data_subset
        self.data_table = []
        self.global_random_state = None
        self.attribute_range = attribute_range
        self.attribute_info = attribute_info
        self.num_target_classes = num_target_classes
        self.global_seed = global_seed
        random.seed(global_seed)
        self.att_perc = attribute_percentage
        self.num_criteria = round(self.att_perc * len(self.attribute_info))
        self.Secure_Aggregation_SMC = secure_aggregation_smc
        self.ToolsForkPPDERT = k_PPD_ERT_functions.ToolsForkPPDERT(self.num_target_classes, self.num_criteria,
                                                                   num_parties, party_id,
                                                                   num_participating_parties,
                                                                   secure_aggregation_parameter_k, spp)

    class Criterion:
        """The criterion objects instantiated from Criterion class are used to divide a data set into two sets"""

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

    def limit_data_indices(self, node_id, branch):
        """RETURNS THE INDICES FOR DATA SAMPLES LIMITED BY node_id and branch
        input: node_id(id of previously created nodes), and branch(True/False)
        output: indices of data sample meet the criterion made by node_id, branch"""

        if node_id == 0:
            limited_data_indices = np.arange(self.data_subset.shape[0])
        else:
            for i in range(len(self.data_table)):
                id_i = np.array(self.data_table[i][0])
                if node_id == id_i:
                    if branch:
                        limited_data_indices = np.array(self.data_table[i][1])  # True branch indices(for node node_id)
                    else:
                        limited_data_indices = np.array(self.data_table[i][2])  # False branch indices(for node node_id)
        return limited_data_indices

    def pick_a_random_split(self, attribute_indices):
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

    def create_criteria(self):
        """ This functions creates a list of criteria since we use the same seed for our random function,
        the created criteria are the same in all the parties and learner in every step
        input: no input
        output: a list containing the generated criteria ({attribute,split} sets)"""

        # SELECT ATTRIBUTES RANDOMLY
        #  number of attributes
        num_att = len(self.attribute_info)
        #  percentage of attributes randomly selected
        att_perc = self.att_perc
        #  number of attributes randomly selected
        num_rnd_att = np.int(round(att_perc * num_att))
        # indices of randomly selected attributes
        rand_attribute_index = [random.randint(0, num_att - 1) for p in range(0, num_rnd_att)]
        # SELECT RANDOM SPLITS
        split_list = self.pick_a_random_split(rand_attribute_index)

        # CREATE A LIST OF {ATTRIBUTE,SPLIT} SETS
        criterion_list = []
        for i in range(num_rnd_att):  # for each selected attribute
            index_of_the_attribute = rand_attribute_index[i]
            attribute_type = self.attribute_info[index_of_the_attribute]
            rnd_split = split_list[i]
            #  make a criterion to evaluate
            criterion_list.append(self.Criterion(attribute_type, index_of_the_attribute, rnd_split))
        return criterion_list

    def check(self, node_id, branch):
        """input: criterion_list, criterion_result_list for excluding the part of the data
        that does not meet the conditions in previous nodes; criterion is the new criterion under scrutiny ;)
        output: labels of the (limited set of) data samples after seperation of samples based on the new criterion"""

        true_set_classes = np.zeros((self.num_criteria, self.num_target_classes))
        false_set_classes = np.zeros((self.num_criteria, self.num_target_classes))
        rnd_sum_others = np.zeros((self.num_criteria, self.num_target_classes))
        rnd_sum_self= np.zeros((self.num_criteria, self.num_target_classes))

        # check my participation
        # !!! RANDOM SEED AND STATE !!!
        self.global_random_state = random.getstate()
        # get in
        self.ToolsForkPPDERT.identify_participating_parties()
        # get out
        random.seed(self.global_seed)
        random.setstate(self.global_random_state)
        # !!! RANDOM SEED AND STATE !!!
        my_presence = self.ToolsForkPPDERT.check_my_presence(self.ToolsForkPPDERT.participating_parties)
        if my_presence:
            pass
        else:  # check for error
            self.ToolsForkPPDERT.update_seeds_sates()
            return

        # create criteria using create_criteria() func.
        criterion_list = self.create_criteria()

        limited_data_indices = self.limit_data_indices(node_id, branch)
        for i in range(self.num_criteria):
            criterion = criterion_list[i]
            if limited_data_indices.shape[0] != 0:
                for sample in self.data_subset[limited_data_indices, :]:
                    if criterion.check(sample):
                        true_set_classes[i, int(sample[-1])] += 1
                    else:
                        false_set_classes[i, int(sample[-1])] += 1

        # FOR SMC PART
        if self.Secure_Aggregation_SMC:
            # !!! RANDOM SEED AND STATE !!!
            self.global_random_state = random.getstate()
            # get in
            true_set_classes, false_set_classes= \
                self.ToolsForkPPDERT.mask(true_set_classes, false_set_classes)
            # get out
            random.seed(self.global_seed)
            random.setstate(self.global_random_state)
            # !!! RANDOM SEED AND STATE !!!

        return true_set_classes, false_set_classes

    def update_data_table(self, criterion, node_id, branch):
        """UPDATE THE DATA TABLE BASED ON THE LEARNED CRITERIA
        Input: the learned criterion, the limitation from previous criteria introduced by node_id, branch
        no output... ;)"""
        true_set_indices = []
        false_set_indices = []
        limited_data_indices = self.limit_data_indices(node_id, branch)
        if limited_data_indices.shape[0] != 0:
            for index in limited_data_indices:
                sample = self.data_subset[index, :]
                if criterion.check(sample):
                    true_set_indices.append(index)
                else:
                    false_set_indices.append(index)
        self.data_table.append(([node_id + 1], true_set_indices, false_set_indices))

    def reset_party(self):
        self.data_table = []
