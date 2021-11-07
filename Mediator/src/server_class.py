import src.Tree_Elements
import src.Tree_Learning_Requirements
import numpy as np
import timeit


class server:

    def __init__(self, global_seed, attribute_range, attribute_info, num_target_classes,\
                 aggregator_func, parties_update_func, attribute_percentage,\
                 included_parties_indices, Secure_Aggregation_SMC):  # parties_reset_func
        self.attribute_range = attribute_range
        self.attribute_info = attribute_info
        self.num_target_classes = num_target_classes
        # self.aggregator_func = aggregator_func
        # self.parties_update_func = parties_update_func
        self.TLR = src.Tree_Learning_Requirements.Tree_Learning_Requirements(global_seed,\
                    self.attribute_range, self.attribute_info, self.num_target_classes,\
                    aggregator_func, parties_update_func, attribute_percentage,\
                    included_parties_indices,Secure_Aggregation_SMC)
        # self.parties_reset_func = parties_reset_func

    # Making Trees
    def make_tree_group(self, impurity_measure='entropy', num_of_trees=10):
        """MAKE A BUNCH OF TREES
        input: data, data info, impurity measure, desiered number of trees
        output: a list incorporating num_of_trees learned trees"""

        tree_group = []
        start_all = timeit.default_timer()
        for i in range(num_of_trees):
            with open('results.txt', 'a+') as file:
                print("Learning tree:", i + 1, "/", num_of_trees)
                file.write('Learning tree: {} / {} \n'.format(str(i+1), str(num_of_trees)))
                start = timeit.default_timer()
                tree_group.append(self.grow_tree(impurity_measure=impurity_measure))
                # self.parties_reset_func()
                stop = timeit.default_timer()
                print("number of secure aggregations (cumulative): ",self.TLR.num_transactions)
                file.write('number of secure aggregations (cumulative): {} \n'.format(str(self.TLR.num_transactions)))
                print("number of updates (cumulative): ", self.TLR.num_updates)
                file.write('number of updates (cumulative): {} \n'.format(str(self.TLR.num_updates)))
                print("Elapsed time: ", stop - start, " Sec")
                file.write('Elapsed time: {} \n'.format(str(stop - start)))
                print("==============================")
                file.write('------------------------------------ \n')

        stop_all = timeit.default_timer()
        with open('results.txt', 'a+') as file:

            print("Elapsed time for learning all trees: ", stop_all - start_all, " Sec")

            file.write("Elapsed time for learning all trees: {} Sec\n".format(str(stop_all - start_all)))

            print("Average elapsed time for learning a tree: ", (stop_all - start_all) / num_of_trees, " Sec")

            file.write("Average elapsed time for learning a tree: {} Sec\n".format(str(stop_all - start_all)))

        return tree_group

    def grow_tree(self, impurity_measure='entropy', node_id=0, branch=None):
        """A recursive function for growing a decision tree by learning from data
        input: a set of data
        output: a learnt tree based on the input data"""

        purity_val, criterion, classes, node_id = \
            self.TLR.find_best_attribute(impurity_measure, node_id,branch)

        if purity_val == 0:
            return src.Tree_Elements.End_Node(
                np.argmax(np.array(classes)))  # the label would be the index of the maximum value in the classes vector
        else:

            # the result of grow_tree() can be an end node (leaf)
            # or a node (tree). Thus, true_branch and false_branch
            # can be either leaf or tree.

            branch = True
            true_branch = self.grow_tree(impurity_measure, node_id, branch)
            branch = False
            false_branch = self.grow_tree(impurity_measure, node_id, branch)

            node = src.Tree_Elements.Node(criterion, true_branch, false_branch)
            return node


