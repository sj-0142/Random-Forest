'''
Explanation for OOPS in python:
    class is like a blueprint which can be copied indefinitely
    constructor __init__ is used to give a different input everytime u create the class to provide more data

    self is the class object itself


ENTROPY AND INFORMATION GAIN:
    H(P1) = p1log2(p1) + (1-p1)log2(1-p1)
    IG = ROOT NODE ENTROPY  +   Weighted average of childern where weights are number of pure elements by total elements per node


'''

import numpy as np
from collections import Counter

class Node:
    def __init__(self,feature = None,threshold = None,left = None,right = None,*, value = None): #asterisk is used so that it becomes compulsotry to say value = 1 everytime we deifne value (easier to differentiante between leaf and non leaf)
        self.feature = feature      #which feature this node was divided with
        self.threshold = threshold 
        self.left = left        #left and right nodes that we are pointing to
        self.right = right
        self.value=value #if value exists it is a leaf node otherwise it is None

    def is_leaf_node(self):
        return self.value is not None


class DecisionTree:

    def __init__(self, min_samples_split = 2, max_depth = 100, n_features = None):
        self.min_samples_split =min_samples_split #minimum samples of impurity for which you would consider splitting 
        self.n_features = n_features    #used when random forest is implemented on DT so we sample n features randomly there
        self.max_depth = max_depth      #stop after reaching this depth 
        self.root = None            #defines a  root for a particular decision tree

    def fit(self, X,y):
        self.n_features = X.shape[1] if not self.n_features else min(X.shape[1], self.n_features)  # if self.n_features is not set, it will use all the features (columns) of X if it is set, then it will take the minimum of the features of X and the specified value
        self.root = self._grow_tree(X,y)
    
    def _grow_tree(self,X,y, depth = 0): #method with underscore means it is an internal method and it is not to be used from the outside

        n_samples, n_feats  = X.shape #returns (rows, columns) so we unpack it into variables
        n_labels = len(np.unique(y)) #gets lenghth of all unique values of y to get all the labels

        #check the stopping criteria
        if(depth>=self.max_depth or n_labels ==1 or n_samples<self.min_samples_split):
            #create a new node and return it
            #since we are stopping the growth of the tree, we still need something to return to the decision maker as the value of the node so we return a new node with the most common value
            leaf_value = self.most_common_label(y)
            return Node(value = leaf_value) 
        

        
        feat_idxs = np.random.choice(n_feats, self.n_features, replace = False) #choose any self.n_features number of features from the total number of n_feats

        #find the best split

        best_feature,best_threshold  = self._best_split(X,y,feat_idxs)
        #create child nodes

        left_idxs, right_idxs  = self._split(X[:,best_feature], best_threshold)
        left = self._grow_tree(X[left_idxs, :], y[left_idxs], depth+1)
        right = self._grow_tree(X[right_idxs, :], y[right_idxs], depth+1)
        return Node(best_feature, best_threshold, left, right)

    def _best_split(self, X,y,feat_idxs):
        best_gain = -1
        split_idx, split_threshold = None

        for feat_idx in feat_idxs:
            X_column = X[:, feat_idx]   
            thresholds = np.unique(X_column)

            for thr in thresholds:
                #calculate the information gain
                gain = self._information_gain(y, X_column, thr)


                if gain>best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = thr

        return split_idx, split_threshold

    def _information_gain(self, y, X_column, threshold):
        pass
        #parent entropy
        parent_entropy = self._entropy(y)

        #create children
        left_idxs, right_idxs = self._split(X_column, threshold)

        if len(left_idxs)==0 or len(right_idxs)==0:
            return 0

        #calcualte the weighted entropy of children

        n = len(y)
        n_l, n_r = len(left_idxs), len(right_idxs)
        e_l, e_r = self._entropy(y[left_idxs]), self._entropy(y[right_idxs])

        child_entropy = (n_l/n) * e_l + (n_r/n) * e_r

        #calculate the Information Gain

        information_gain = parent_entropy - child_entropy

        return information_gain

    def _split(self, X_column, split_thresh):
        left_idxs = np.argwhere(X_column<=split_thresh).flatten() #gives all the element indices that follow that criteria in the form of a list of list (2d List) so we need to use flatten () to make it a single list

        right_idxs = np.argwhere(X_column>split_thresh).flatten()
        return left_idxs, right_idxs


    def _entropy(self, y):
        hist = np.bincount(y) #histogram showing frequencies of all the elements (basically a hashmap where indices are mapped to their frequencies) 

        ps = hist/len(y) 
        return -np.sum([p*np.log2(p)for p in ps if p>0]) 


    def _most_common_label(self, y): #returns the most common label of Y
         counter = Counter(y)
         return counter.most_common(1)[0][0]
    

    def predict(self, X):
        return np.array([self._traverse_tree(x)for x in X])
    
    def _traverse_tree(self, x, node):
        if node.is_leaf_node():
            return node.value
        
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left) #if condition is satisfied we go to left side or else we go to right side
        return self._traverse_tree(x,node.right)