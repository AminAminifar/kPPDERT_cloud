import numpy as np
import random
import copy


class ToolsForkPPDERT:
    def __init__(self, num_target_classes, num_criteria, num_parties, party_id,
                 num_participating_parties, secure_aggregation_parameter_k, spp):
        self.num_target_classes = num_target_classes
        self.num_criteria = num_criteria
        self.num_parties = num_parties
        self.party_ID = party_id
        self.num_participating_parties = num_participating_parties
        self.participating_parties = []
        self.Secure_Aggregation_Parameter_k = secure_aggregation_parameter_k
        self.SPP = spp
        self.SPP_state = None
        self.SSP_self = None
        self.SSP_others = []
        self.SSA_self = []
        self.SSA_others = []
        self.SSP_self_state = None
        self.SSP_others_state = []#None
        self.SSA_self_state = []#None
        self.SSA_others_state = []#None
        self.set_self_seeds()

    def set_self_seeds(self):
        self.SSP_self = np.random.randint(10 ** 8, size=1)[0]
        for i in range(0, self.num_parties):  # one more but not used
            self.SSA_self.append(np.random.randint(10 ** 8, size=1)[0])
            self.SSA_self_state.append(None)

    def identify_participating_parties(self):
        """identify which parties will participate in this round
         for selecting the best candidate node/leaf
         input: no input
         output: IDs (indices) of participating parties"""

        parties = list(range(0, self.num_parties))

        # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
        random.seed(self.SPP)
        if self.SPP_state is not None:
            random.setstate(self.SPP_state)
        random.shuffle(parties)
        self.SPP_state = random.getstate()
        # !!! USING PARTICULAR RANDOM SEED AND STATE !!!

        participating_parties_ids = parties[0: self.num_participating_parties]
        self.participating_parties = np.sort(participating_parties_ids)

        return

    def exclude_my_id(self, party_ids):
        """I my ID is among the received IDs, this function will remove it
        Input: party_IDs
        Output: party_IDs with no self.party_ID in it"""

        if any(party_ids == self.party_ID):
            index = np.where(party_ids == self.party_ID)[0][0]
            party_ids = np.delete(party_ids, index)

        return party_ids

    def check_my_presence(self, party_ids):
        """Check if my ID is among the received IDs
        Input: party_IDs
        Output: a flag showing my presence state"""

        my_presence = False
        if any(party_ids == self.party_ID):
            my_presence = True

        return my_presence

    def identify_parties_self(self):
        """identifying the peer parties of mine for secure aggregation.
        identifying for which parties I am among the peer parties for secure aggregation
        input: participating parties in this round
        output: IDs (indices) of Peer parties, and a Flag that shows I was included in peer parties"""
        participating_parties_temp = copy.deepcopy(self.participating_parties)
        participating_parties_temp = self.exclude_my_id(participating_parties_temp)

        # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
        random.seed(self.SSP_self)
        if self.SSP_self_state is not None:
            random.setstate(self.SSP_self_state)
        random.shuffle(participating_parties_temp)
        self.SSP_self_state = random.getstate()
        # !!! USING PARTICULAR RANDOM SEED AND STATE !!!

        peer_parties = participating_parties_temp[0: self.Secure_Aggregation_Parameter_k + 1]
        peer_parties = np.array(peer_parties)
        peer_parties = np.sort(peer_parties)

        return peer_parties

    def identify_parties_others(self):
        """identifying the peer parties of others for secure aggregation, and
        if I need to participate in their secure aggregation by generating random masks
        input: participating parties in this round"""

        peer_parties = []
        for ID in self.participating_parties:
            participating_parties_temp = copy.deepcopy(self.participating_parties)

            # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
            random.seed(self.SSP_others[ID])
            if self.SSP_others_state[ID] is not None:
                random.setstate(self.SSP_others_state[ID])
            random.shuffle(participating_parties_temp)
            self.SSP_others_state[ID] = random.getstate()
            # !!! USING PARTICULAR RANDOM SEED AND STATE !!!

            participating_parties_temp = participating_parties_temp[0: self.Secure_Aggregation_Parameter_k + 1]
            if self.check_my_presence(participating_parties_temp):
                peer_parties.append(ID)

        peer_parties = np.array(peer_parties)
        return peer_parties

    def generate_and_aggregate_random_masks(self, party_ids, mask_type):
        """generates random masks based on the received IDs, seeds, and states
        Input: party_IDs, mask_type= 'self' or 'others'
        Output: rnd_sum"""

        rnd_sum = np.zeros((self.num_criteria, self.num_target_classes))
        max_val = 10 ** 7  # this can be changed (by user)

        if mask_type == 'self':
            for ID in party_ids:
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
                random.seed(self.SSA_self[ID])

                if self.SSA_self_state[ID] is not None:
                    random.setstate(self.SSA_self_state[ID])

                rnd_sum += [[random.randint(0, max_val) for p in range(0, self.num_target_classes)]
                            for q in range(0, self.num_criteria)]

                self.SSA_self_state[ID] = random.getstate()
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
        elif mask_type == 'others':
            for ID in party_ids:
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!
                random.seed(self.SSA_others[ID])

                if self.SSA_others_state[ID] is not None:
                    random.setstate(self.SSA_others_state[ID])

                rnd_sum += [[random.randint(0, max_val) for p in range(0, self.num_target_classes)]
                            for q in range(0, self.num_criteria)]

                self.SSA_others_state[ID] = random.getstate()
                # !!! USING PARTICULAR RANDOM SEED AND STATE !!!

        return rnd_sum

    def generate_mask(self):
        """generate masks based on the seeds
        Input: no input
        Output: masks to be used mask my secret values"""

        # identify my peer parties for Secure Aggregation
        identified_parties_self = self.identify_parties_self()
        identified_parties_self = self.exclude_my_id(identified_parties_self)

        # calculate rnd_sum_self
        rnd_sum_self = self.generate_and_aggregate_random_masks(party_ids=identified_parties_self, mask_type='self')

        # identify parties which I need to collaborate for Secure Aggregation
        identified_parties_others = self.identify_parties_others()
        identified_parties_others = self.exclude_my_id(identified_parties_others)

        # calculate rnd_sum_others
        rnd_sum_others = \
            self.generate_and_aggregate_random_masks(party_ids=identified_parties_others, mask_type='others')


        return rnd_sum_self, rnd_sum_others

    def mask(self, true_set_classes, false_set_classes):
        """generates and adds the masks to the received values
        input: true_set_classes, false_set_classes
        Output: masked true_set_classes, false_set_classes"""

        rnd_sum_self, rnd_sum_others = self.generate_mask()

        # TO Be Changed
        true_set_classes += rnd_sum_others
        false_set_classes += rnd_sum_others
        true_set_classes -= rnd_sum_self
        false_set_classes -= rnd_sum_self
        return true_set_classes, false_set_classes

    def update_seeds_sates(self):
        """To update the random function's state for different seeds when I am not participating in this round
        input: no input
        output: no output"""

        parties = list(range(0, self.num_parties))
        # to update random function's state for SSP_self
        _ = self.identify_parties_self()

        # to update random function's state for SSP_others
        _ = self.identify_parties_others()
