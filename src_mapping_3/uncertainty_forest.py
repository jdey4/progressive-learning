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
                np.zeros((self._feature_dimension,1),dtype=float),
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
            def __init__(self, estimators, estimators_samples_, classes, tree_id_to_leaf_profile, parallel, n_jobs):
                self.estimators = estimators
                self.n_estimators = len(estimators_samples_)
                self.classes_ = classes
                self.tree_id_to_leaf_profile = tree_id_to_leaf_profile
                self.parallel = parallel
                self.estimators_samples_ = estimators_samples_
                self.n_jobs = n_jobs
            
            def fit(self, nodes_across_trees=None, y=None, voters_to_be_mapped=None, current_task_voters=None, fitting = False, map=False):
                self.tree_idx_to_node_ids_to_posterior_map = {}
                self.tree_idx_to_node_ids_to_sample_count_map = {}

                if map == False:
                    def worker(tree_idx):
                        nodes = nodes_across_trees[tree_idx]
                        oob_samples = np.delete(range(len(nodes)), self.estimators_samples_[tree_idx])
                        cal_nodes = nodes #nodes[oob_samples] if fitting else nodes
                        y_cal = y #y[oob_samples] if fitting else y                    
                        
                        #create a map from the unique node ids to their classwise posteriors
                        node_ids_to_posterior_map = {}
                        node_ids_to_sample_count_map = {}

                        #fill in the posteriors 
                        for node_id in np.unique(cal_nodes):
                            cal_idxs_of_node_id = np.where(cal_nodes == node_id)[0]
                            cal_ys_of_node = y_cal[cal_idxs_of_node_id]
                            class_counts = [len(np.where(cal_ys_of_node == y)[0]) for y in np.unique(y) ]
                            sample_no = np.sum(class_counts)
                            posteriors = np.nan_to_num(np.array(class_counts) / sample_no)

                            #finite sample correction
                            posteriors_corrected = _finite_sample_correction(posteriors, len(cal_idxs_of_node_id), len(self.classes_))
                            node_ids_to_posterior_map[node_id] = posteriors_corrected
                            node_ids_to_sample_count_map[node_id] = sample_no

                        #add the node_ids_to_posterior_map to the overall tree_idx map 
                        self.tree_idx_to_node_ids_to_posterior_map[tree_idx] = node_ids_to_posterior_map
                        self.tree_idx_to_node_ids_to_sample_count_map[tree_idx] = node_ids_to_sample_count_map
                        
                    for tree_idx in range(self.n_estimators):
                            worker(tree_idx)
                    return self
                else:
                    node_ids_to_posterior_map = {}
                    _leaf_posteriors = []
                    _leaf_sample_covered = []
                    _leaf_current_task_data_correction = []

                    def worker(
                        node, 
                        feature, children_left, 
                        children_right, 
                        threshold, 
                        mul, 
                        profile_mat, 
                        posterior_map, 
                        sample_map,
                        profile_map,
                        current_task_posterior_map,
                        current_task_sample_map,
                        current_task_mul
                        ):
                        if children_left[node] == children_right[node]:
                            
                            if node in list(posterior_map.keys()):
                                k1 = np.prod(
                                    (profile_mat[:,1]-profile_mat[:,0])/(profile_map[node][:,1]- profile_map[node][:,0])
                                )
                                #print(profile_mat, profile_map[node], 'kukuta')
                                #print(mul, 'kutta')
                                #print(node, current_task_posterior_map,'osovyo')

                                if node in list(current_task_posterior_map.keys()):
                                    current_task_cls_sample_count = current_task_posterior_map[node]*current_task_sample_map[node]
                                    total_current_task_cls = len(current_task_mul)
                                    individual_count = sample_map[node]*posterior_map[node]
                                    correction = np.zeros(len(individual_count), dtype=float)

                                    for idx, cls_count in enumerate(list(individual_count)):
                                        #print(np.sum(((cls_count*mul*current_task_mul)/current_task_cls_sample_count)/total_current_task_cls), correction.shape)
                                        correction[idx] = np.sum(((cls_count*mul*current_task_mul)/current_task_cls_sample_count)/total_current_task_cls)

                                    #print(correction)
                                    _leaf_posteriors.append(
                                        k1*sample_map[node]*posterior_map[node]*correction
                                    )
                                    _leaf_sample_covered.append(
                                        k1*sample_map[node]*np.sum(correction)
                                    )

                                else:
                                    _leaf_posteriors.append(
                                        k1*sample_map[node]*posterior_map[node]
                                    )
                                    _leaf_sample_covered.append(
                                        k1*sample_map[node]
                                    )
                                
                                '''target_task = voter_sample_map[2].tree_dx
                                current_task_data_correction = '''

                            
                        else:
                            profile_mat_left = profile_mat.copy()
                            profile_mat_right = profile_mat.copy()
                            current_feature = feature[node]
                            current_threshold = threshold[node]
                            feature_range = profile_mat[current_feature]

                            if current_threshold>feature_range[0] and current_threshold<feature_range[1]:
                                profile_mat_left[current_feature][1] = current_threshold
                                profile_mat_right[current_feature][0] = current_threshold
                                mul_left = mul*(
                                    current_threshold - feature_range[0]
                                    )/(
                                        feature_range[1] - feature_range[0]
                                    )
                                mul_right = mul*(
                                    feature_range[1] - current_threshold
                                    )/(
                                        feature_range[1] - feature_range[0]
                                    )

                                
                                worker(
                                    children_left[node],
                                    feature,
                                    children_left,
                                    children_right,
                                    threshold,
                                    mul_left,
                                    profile_mat_left,
                                    posterior_map,
                                    sample_map,
                                    profile_map,
                                    current_task_posterior_map,
                                    current_task_sample_map,
                                    current_task_mul
                                )
                                worker(
                                    children_right[node],
                                    feature,
                                    children_left,
                                    children_right,
                                    threshold,
                                    mul_right,
                                    profile_mat_right,
                                    posterior_map,
                                    sample_map,
                                    profile_map,
                                    current_task_posterior_map,
                                    current_task_sample_map,
                                    current_task_mul
                                )
                            elif current_threshold <= feature_range[0]:
                                
                                return worker(
                                    children_right[node],
                                    feature,
                                    children_left,
                                    children_right,
                                    threshold,
                                    mul,
                                    profile_mat_right,
                                    posterior_map,
                                    sample_map,
                                    profile_map,
                                    current_task_posterior_map,
                                    current_task_sample_map,
                                    current_task_mul
                                )
                            elif current_threshold >= feature_range[1]:
                                
                                return worker(
                                    children_left[node],
                                    feature,
                                    children_left,
                                    children_right,
                                    threshold,
                                    mul,
                                    profile_mat_left,
                                    posterior_map,
                                    sample_map,
                                    profile_map,
                                    current_task_posterior_map,
                                    current_task_sample_map,
                                    current_task_mul
                                )

                    def map_leaf(voters_to_be_mapped,leaf,profile,current_task_mul):
                        #print(leaf,'leaf id')
                        #print(voters_to_be_mapped)
                        for ids, current_voter in enumerate(voters_to_be_mapped):
                            estimators = current_voter.estimators
                            print(estimators[0].tree_, 'eije ami asi')

                            posteriors_to_be_mapped = current_voter.tree_idx_to_node_ids_to_posterior_map
                            sample_count_map = current_voter.tree_idx_to_node_ids_to_sample_count_map
                            profile_map = current_voter.tree_id_to_leaf_profile

                            corresponding_current_task_voter = current_task_voters[ids]
                            corresponding_current_task_voter_sample_map = corresponding_current_task_voter.tree_idx_to_node_ids_to_sample_count_map
                            corresponding_current_task_voter_posterior_map = corresponding_current_task_voter.tree_idx_to_node_ids_to_posterior_map

                            #print(posteriors_to_be_mapped,'modon',corresponding_current_task_voter_posterior_map,'hello')
                            posterior = np.zeros(
                                (len(estimators), len(current_voter.classes_)), 
                                dtype = float
                            )

                            for tree_id,tree in enumerate(estimators):
                                feature = tree.tree_.feature
                                children_left = tree.tree_.children_left
                                children_right = tree.tree_.children_right
                                threshold = tree.tree_.threshold

                                worker(
                                        0,
                                        feature,
                                        children_left,
                                        children_right,
                                        threshold,
                                        1,
                                        profile,
                                        posteriors_to_be_mapped[tree_id],
                                        sample_count_map[tree_id],
                                        profile_map[tree_id],
                                        corresponding_current_task_voter_posterior_map[tree_id],
                                        corresponding_current_task_voter_sample_map[tree_id],
                                        current_task_mul
                                        )
                                
                                num = np.array(_leaf_posteriors)
                                den = np.array(_leaf_sample_covered)
                                posterior[tree_id] = np.sum(
                                    num, axis=0
                                ) / np.sum(den)

                                _leaf_posteriors.clear()
                                _leaf_sample_covered.clear()
                                _leaf_current_task_data_correction.clear()

                                #print(posterior[tree_id],'kukuta')
                            if ids == 0:
                                posterior_ = posterior
                            else:
                                posterior_ = np.concatenate(
                                    (
                                        posterior_,posterior
                                    ),
                                    axis=0
                                )
                        node_ids_to_posterior_map[leaf] = np.mean(
                            posterior_, axis=0
                        )

                    #################################################################################   
                    tree_idx = list(self.tree_id_to_leaf_profile.keys())
                    node_ids_to_posterior_map = {}
                    current_task_new_voter_sample_count = current_task_voters[-1].tree_idx_to_node_ids_to_sample_count_map
                    current_task_new_voter_posterior_map = current_task_voters[-1].tree_idx_to_node_ids_to_posterior_map

                    for idx in tree_idx: 
                        leaf_id = list(self.tree_id_to_leaf_profile[idx].keys())

                        for leaf in leaf_id:
                            #print(leaf,'fervebgtr')
                            current_task_mul = current_task_new_voter_sample_count[idx][leaf]*current_task_new_voter_posterior_map[idx][leaf]
                            profile = self.tree_id_to_leaf_profile[idx][leaf]
                            
                            map_leaf(
                                voters_to_be_mapped,
                                leaf,
                                profile,
                                current_task_mul
                                )
                            #print(profile,'profile',leaf, idx)
                        #print(node_ids_to_posterior_map,'jerhubiruu')
                        self.tree_idx_to_node_ids_to_posterior_map[idx] = node_ids_to_posterior_map
                        #node_ids_to_posterior_map.clear()
                    
                    #print(self.tree_idx_to_node_ids_to_posterior_map,'hello')
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
        self.voter = Voter(
            estimators = self.ensemble.estimators_,
            estimators_samples_ = self.ensemble.estimators_samples_, 
            classes = self.classes_, 
            tree_id_to_leaf_profile = self.tree_id_to_leaf_profile,
            parallel = self.parallel,
            n_jobs = self.n_jobs
            )
        self.voter.fit(
            nodes_across_trees = nodes_across_trees, 
            y=y, 
            fitting = True
            )
        self.fitted = True


    def predict(self, X):
        return self.classes_[np.argmax(self.predict_proba(X), axis=-1)]

    def predict_proba(self, X):
        return self.voter.predict_proba(self.transform(X))
