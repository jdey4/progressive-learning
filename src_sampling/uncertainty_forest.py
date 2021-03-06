'''
Primary Author: Will LeVine
Email: levinewill@icloud.com
'''

#Model
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier

#Infrastructure
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import NotFittedError

#Data Handling
from sklearn.utils.validation import (
    check_X_y,
    check_array,
    NotFittedError,
)
from sklearn.utils.multiclass import check_classification_targets

#Utils
from joblib import Parallel, delayed
import numpy as np

def _finite_sample_correction(posteriors, num_points_in_partition, num_classes):
    '''
    encourage posteriors to approach uniform when there is low data
    '''
    correction_constant = 1 / (num_classes * num_points_in_partition)

    zero_posterior_idxs = np.where(posteriors == 0)[0]
    posteriors[zero_posterior_idxs] = correction_constant
    
    posteriors /= sum(posteriors)
    
    return posteriors

class UncertaintyForest(BaseEstimator, ClassifierMixin):
    '''
    based off of https://arxiv.org/pdf/1907.00325.pdf
    '''
    def __init__(
        self,
        max_depth=30,
        min_samples_leaf=1,
        max_samples = 0.63,
        max_features_tree = "auto",
        n_estimators=100,
        bootstrap=False,
        parallel=True,
        n_jobs = None):

        #Tree parameters.
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.max_features_tree = max_features_tree

        #Bag parameters
        self.n_estimators = n_estimators
        self.bootstrap = bootstrap
        self.max_samples = max_samples

        #Model parameters.
        self.parallel = parallel
        if self.parallel and n_jobs == None:
            self.n_jobs = self.n_estimators
        else:
            self.n_jobs = n_jobs
        self.fitted = False

    def _check_fit(self):
        '''
        raise a NotFittedError if the model isn't fit
        '''
        if not self.fitted:
                msg = (
                        "This %(name)s instance is not fitted yet. Call 'fit' with "
                        "appropriate arguments before using this estimator."
                )
                raise NotFittedError(msg % {"name": type(self).__name__})

    def transform(self, X):
        '''
        get the estimated posteriors across trees
        '''
        X = check_array(X)
                
        def worker(tree_idx, tree):
            #get the nodes of X
            # Drop each estimation example down the tree, and record its 'y' value.
            return tree.apply(X)
            

        if self.parallel:
            return np.array(
                    Parallel(n_jobs=self.n_jobs)(
                            delayed(worker)(tree_idx, tree) for tree_idx, tree in enumerate(self.ensemble.estimators_)
                    )
            )         
        else:
            return np.array(
                    [worker(tree_idx, tree) for tree_idx, tree in enumerate(self.ensemble.estimators_)]
                    )
    
    # function added to do partition mapping
    def _profile_leaf(self):
        self.tree_id_to_leaf_profile = {}
        leaf_profile = {}
        #print('hi')
        def worker(node, children_left, children_right, feature, threshold, profile_mat):

            if children_left[node] == children_right[node]:
                profile_mat_ = profile_mat.copy()
                leaf_profile[node] = profile_mat_
                #print(node,'nodes')
            else:
                feature_indx = feature[node]
                profile_mat_ = profile_mat.copy()
                profile_mat_[feature_indx,1] = threshold[node]

                worker(
                    children_left[node], 
                    children_left, 
                    children_right, 
                    feature, 
                    threshold,
                    profile_mat_
                    )
                        
                profile_mat_ = profile_mat.copy()
                profile_mat_[feature_indx,0] = threshold[node]
                worker(
                    children_right[node], 
                    children_left, 
                    children_right, 
                    feature, 
                    threshold,
                    profile_mat_
                    )

        profile_mat = np.concatenate(
            (
                -1*np.ones((self._feature_dimension,1),dtype=float),
                np.ones((self._feature_dimension,1),dtype=float)
            ),
            axis = 1
        )

        for tree_id, estimator in enumerate(self.ensemble.estimators_):
            leaf_profile = {}
            feature = estimator.tree_.feature
            children_left = estimator.tree_.children_left
            children_right = estimator.tree_.children_right
            threshold = estimator.tree_.threshold
            #print(children_left,children_right)

            worker(
                    0, 
                    children_left, 
                    children_right, 
                    feature, 
                    threshold,
                    profile_mat.copy()
                    )

            self.tree_id_to_leaf_profile[tree_id] = leaf_profile
        #print(self.tree_id_to_leaf_profile,'gdgfg')


    def get_transformer(self):
        return lambda X : self.transform(X)
        
    def vote(self, nodes_across_trees):
        return self.voter.predict(nodes_across_trees)
        
    def get_voter(self):
        return self.voter
        
                        
    def fit(self, X, y):

        #format X and y
        X, y = check_X_y(X, y)
        check_classification_targets(y)
        self.classes_, y = np.unique(y, return_inverse=True)
        
        #define the ensemble
        self.ensemble = BaggingClassifier(
            DecisionTreeClassifier(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features_tree
            ),
            n_estimators=self.n_estimators,
            max_samples=self.max_samples,
            bootstrap=self.bootstrap,
            n_jobs = self.n_jobs
        )
        
        #fit the ensemble
        self.ensemble.fit(X, y)
        self._feature_dimension = X.shape[1]
        #profile trees for partition mapping
        self._profile_leaf()

        class Voter(BaseEstimator):
            def __init__(self, estimators_samples_, tree_id_to_leaf_profile, classes, parallel, n_jobs):
                self.n_estimators = len(estimators_samples_)
                self.classes_ = classes
                self.parallel = parallel
                self.estimators_samples_ = estimators_samples_
                self.tree_id_to_leaf_profile = tree_id_to_leaf_profile
                self.n_jobs = n_jobs
            
            def fit(self, nodes_across_trees, y, tree_id_to_leaf_profile=None, fitting = False, mapper=False):

                if tree_id_to_leaf_profile != None:
                    self.tree_id_to_leaf_profile = tree_id_to_leaf_profile

                self.tree_idx_to_node_ids_to_posterior_map = {}

                def worker(tree_idx):
                    nodes = nodes_across_trees[tree_idx]
                    oob_samples = np.delete(range(len(nodes)), self.estimators_samples_[tree_idx])
                    cal_nodes = nodes#[oob_samples] if fitting else nodes
                    y_cal = y#[oob_samples] if fitting else y                    
                    all_nodes = np.array(list(self.tree_id_to_leaf_profile[tree_idx].keys()))
                    #print(all_nodes,'kkukkuta')
                    #create a map from the unique node ids to their classwise posteriors
                    node_ids_to_posterior_map = {}

                    #fill in the posteriors 
                    if mapper == True:
                        for node_id in np.unique(cal_nodes):
                            cal_idxs_of_node_id = np.where(cal_nodes == node_id)[0]
                            cal_ys_of_node = y_cal[cal_idxs_of_node_id]
                            
                            if len(cal_ys_of_node) != 1:
                                posteriors = np.nan_to_num(np.mean(cal_ys_of_node,axis=0))
                            else:
                                posteriors = cal_ys_of_node

                            #print(posteriors,'hlw')
                            #finite sample correction      
                            posteriors_corrected = _finite_sample_correction(posteriors, 1, len(self.classes_))
                            node_ids_to_posterior_map[node_id] = posteriors_corrected
                    else:
                        for node_id in np.unique(all_nodes):
                            cal_idxs_of_node_id = np.where(cal_nodes == node_id)[0]
                            cal_ys_of_node = y_cal[cal_idxs_of_node_id]
                            class_counts = [len(np.where(cal_ys_of_node == y)[0]) for y in np.unique(y) ]
                            sample_no = np.sum(class_counts)

                            if sample_no != 0:
                                posteriors = np.nan_to_num(np.array(class_counts) / sample_no)
                            else:
                                posteriors = np.zeros(len(self.classes_),dtype=float)

                            #finite sample correction
                            total_samples = len(cal_idxs_of_node_id)

                            if total_samples == 0:
                                total_samples = 1
                                    
                            posteriors_corrected = _finite_sample_correction(posteriors, total_samples, len(self.classes_))
                            node_ids_to_posterior_map[node_id] = posteriors_corrected

                    #add the node_ids_to_posterior_map to the overall tree_idx map 
                    self.tree_idx_to_node_ids_to_posterior_map[tree_idx] = node_ids_to_posterior_map
                    
                for tree_idx in range(self.n_estimators):
                        worker(tree_idx)
                return self
                        
                        
            def predict_proba(self, nodes_across_trees):
                def worker(tree_idx):
                    #get the node_ids_to_posterior_map for this tree
                    node_ids_to_posterior_map = self.tree_idx_to_node_ids_to_posterior_map[tree_idx]

                    #get the nodes of X
                    nodes = nodes_across_trees[tree_idx]

                    posteriors = []
                    node_ids = node_ids_to_posterior_map.keys()

                    #loop over nodes of X
                    for node in nodes:
                        #if we've seen this node before, simply get the posterior
                        if node in node_ids:
                            posteriors.append(node_ids_to_posterior_map[node])
                        #if we haven't seen this node before, simply use the uniform posterior 
                        else:
                            posteriors.append(np.ones((len(np.unique(self.classes_)))) / len(self.classes_))
                    return posteriors

                if self.parallel:
                    return np.mean(
                            Parallel(n_jobs=self.n_jobs)(
                                    delayed(worker)(tree_idx) for tree_idx in range(self.n_estimators)
                            ), axis = 0
                    )

                else:
                    return np.mean(
                            [worker(tree_idx) for tree_idx in range(self.n_estimators)], axis = 0)
                
        #get the nodes of the calibration set
        nodes_across_trees = self.transform(X) 
        self.voter = Voter(estimators_samples_ = self.ensemble.estimators_samples_, tree_id_to_leaf_profile = self.tree_id_to_leaf_profile, classes = self.classes_, parallel = self.parallel, n_jobs = self.n_jobs)
        self.voter.fit(nodes_across_trees, y, fitting = True)
        self.fitted = True

    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=-1)]

    def predict_proba(self, X):
        return self.voter.predict_proba(self.transform(X))
