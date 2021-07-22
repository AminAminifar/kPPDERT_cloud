#_Tree Elements
class End_Node:
    def __init__(self, in_label):
        self.label = in_label

    def __del__(self):
        pass


class Node:
    def __init__(self, criterion, true_branch, false_branch):
        self.criterion = criterion
        self.true_branch = true_branch
        self.false_branch = false_branch

    def __del__(self):
        pass