import random
import numpy as np
import select
import pickle


class Interface:

    def __init__(self, clients, socket_list):
        self.clients = clients
        self.socket_list = socket_list
        self.HEADER_LENGTH = 10


    # Functions working with instantiated objects from party class
    def aggregator(self, node_id, branch, num_target_classes, num_criteria):
        """aggregates the results from all parties for desired criterion
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

        # print("server is sending check for each party")
        for party in self.clients:
            message = {"flag": "check", "node_id": node_id, "branch": branch}

            party.send(pickle.dumps(message))
            # print("Send check")

        i = 0
        flag = True
        while flag:
            read_sockets, _, exception_sockets = select.select(self.socket_list, [], self.socket_list)

            for notified_socket in read_sockets:
                if i != len(self.clients):
                    i = i + 1
                    message = notified_socket.recv(2048)
                    message = pickle.loads(message)
                    true_temp = message["true_temp"]
                    false_temp = message["false_temp"]
                    true_set_classes += true_temp
                    false_set_classes += false_temp
                if i == len(self.clients):
                    flag = False

        # print("check process is completed")
        return true_set_classes, false_set_classes

    def parties_update(self, best_criterion, node_id, branch):
        """Update the data table of all the parties after picking a criterion for our tree
        input: chosen criterion, node_id and branch of previous node
        (this new criterion adds new limitation on samples after previous limitation identified by node_id and branch)
        no output; just updates a list in every party"""

        # print("server is sending update for each party")

        for party in self.clients:
            message = {"flag": "update_data_table",
                       "attribute_type": best_criterion.attribute_type,
                       "attribute_index": best_criterion.attribute_index,
                       "point_or_category": best_criterion.point_or_category,
                       "node_id": node_id, "branch": branch}

            party.send(pickle.dumps(message))
#             print("Update table sent")

        i = 0
        flag = True
        while flag:
            read_sockets, _, exception_sockets = select.select(self.socket_list, [], self.socket_list)
            for notified_socket in read_sockets:
                if i != len(self.clients):
                    message = notified_socket.recv(2048)
                    i = i + 1
                if i == len(self.clients):
                    flag = False

        # print("update process is completed")
