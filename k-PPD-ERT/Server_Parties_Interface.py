import random
import numpy as np
import time
from sys import getsizeof

class interface:

    def __init__(self,parties, proportion_of_collaborating_parties, num_parties):
        self.parties = parties
        self.proportion_of_collaborating_parties = proportion_of_collaborating_parties
        self.num_parties = num_parties
    
    #_Functions working with instantiated objects from party class
    def aggregator(self, node_id, branch, random_func_state, num_target_classes, num_criteria):
        """aggregates the results from all parties for desiered criterion
        with the specified limitation on data (criterion_list, criterion_result_list).
        input: the father 'node_id', and the 'branch' (true/false), the state of the random
        function, 'random_func_state', to have all the parties in the same state for the
        random function; this is needed here because we are simulating on one machine.
        If we do our computations on several machines there would be no need to set the sate of
        the random function after setting the seed. Finally, the number of to-be-checked criteria,
        'num_criteria', and the number of classes for the data labels, 'num_target_classes', are needed here.
        output: the classes of samples (limited based on previous nodes criteria) after division
        by the new criteria (to-be-checked criteria)."""

        true_set_classes = np.zeros((num_criteria, num_target_classes))
        false_set_classes = np.zeros((num_criteria, num_target_classes))
        for party in self.parties:

            random.setstate(random_func_state)  # It can also be set in the object (instantiated party)
            output = party.check(node_id, branch)
            # print(getsizeof(false_temp)+getsizeof(false_temp)+getsizeof(node_id)+getsizeof(branch));exit()
            if output is not None:
                true_temp, false_temp = output[0], output[1]
                true_set_classes += true_temp
                false_set_classes += false_temp

        return true_set_classes, false_set_classes


    def parties_update(self, best_criterion, node_id, branch):
        """Update the data table of all the parties after picking a criterion for our tree
        input: chosen criterion, node_id and branch of previous node
        (this new criterion adds new limitation on samples after previous limitation identified by node_id and branch)
        no output; just updates a list in every party"""
        for party in self.parties:
            party.update_data_table(best_criterion, node_id, branch)
            # print(getsizeof(best_criterion)+getsizeof(node_id)+getsizeof(branch));exit()

    def party_updates_others(self, party_id, ssp_self, ssa_self):
        """description"""

        for i in range(0,self.num_parties):
            self.parties[i].ToolsForkPPDERT.SSP_others.append(ssp_self)
            self.parties[i].ToolsForkPPDERT.SSP_others_state.append(None)
            self.parties[i].ToolsForkPPDERT.SSA_others.append(ssa_self[i])
            self.parties[i].ToolsForkPPDERT.SSA_others_state.append(None)

    def initialize_parties(self):
        """Exchange random seeds by data holder parties in the initialization phase
        input: no input
        output: no output"""
        for party in self.parties:
            self.party_updates_others(party_id=party.ToolsForkPPDERT.party_ID,
                                      ssp_self=party.ToolsForkPPDERT.SSP_self,
                                      ssa_self=party.ToolsForkPPDERT.SSA_self)

    def my_print(self):
        for party in self.parties:
            print("====================")
            print(party.ToolsForkPPDERT.SSP_self)
            print(party.ToolsForkPPDERT.SSP_others)
            print(party.ToolsForkPPDERT.SSA_self)
            print(party.ToolsForkPPDERT.SSA_others)
            print("====================")
