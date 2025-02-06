import numpy as np

def gini_impurity(y):
    """
    Compute the Gini impurity for a given set of labels.
    """
    unique, counts = np.unique(y, return_counts=True)
    probs = counts / len(y)
    return 1 - np.sum(probs ** 2)

def split_data(X, y, feature_index, threshold):
    """
    Split the data into two parts based on the given feature index and threshold.
    """
    left_mask = X[:, feature_index] <= threshold
    right_mask = ~left_mask
    return X[left_mask], X[right_mask], y[left_mask], y[right_mask]

def best_split(X, y, min_samples_split=15):
    """
    Find the best split that minimizes the Gini impurity.
    """
    best_gini = float('inf')
    best_split_info = None
    
    # Consider all features
    for feature_index in range(X.shape[1]):
        feature_values = np.unique(X[:, feature_index])
        
        for threshold in feature_values:
            X_left, X_right, y_left, y_right = split_data(X, y, feature_index, threshold)
            
            if len(y_left) < min_samples_split or len(y_right) < min_samples_split:
                continue
            
            gini_left = gini_impurity(y_left)
            gini_right = gini_impurity(y_right)
            
            gini = (len(y_left) / len(y)) * gini_left + (len(y_right) / len(y)) * gini_right
            
            if gini < best_gini:
                best_gini = gini
                best_split_info = {
                    'feature_index': feature_index,
                    'threshold': threshold,
                    'left': (X_left, y_left),
                    'right': (X_right, y_right)
                }
    
    return best_split_info

def build_tree(X, y, depth=0, max_depth=7, min_samples_split=15):
    """
    Recursively build a decision tree.
    """
    if depth >= max_depth or len(np.unique(y)) == 1 or len(y) < min_samples_split:
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    split_info = best_split(X, y, min_samples_split)
    
    if split_info is None:
        unique, counts = np.unique(y, return_counts=True)
        return unique[np.argmax(counts)]
    
    left_tree = build_tree(
        split_info['left'][0], 
        split_info['left'][1], 
        depth + 1, 
        max_depth, 
        min_samples_split
    )
    right_tree = build_tree(
        split_info['right'][0], 
        split_info['right'][1], 
        depth + 1, 
        max_depth, 
        min_samples_split
    )
    
    return {
        'feature_index': split_info['feature_index'],
        'threshold': split_info['threshold'],
        'left': left_tree,
        'right': right_tree
    }

def predict_sample(tree, sample):
    """
    Predict the class for a single sample.
    """
    while isinstance(tree, dict):
        if sample[tree['feature_index']] <= tree['threshold']:
            tree = tree['left']
        else:
            tree = tree['right']
    return tree

def predict(tree, X):
    """
    Predict the classes for a set of samples.
    """
    if hasattr(X, 'toarray'):
        X = X.toarray()
    return np.array([predict_sample(tree, sample) for sample in X])

def decision_tree_classifier(X_train, X_test, y_train, y_test, max_depth=7, 
                           min_samples_split=15, return_predictions=False,
                           pruning=False):
    """
    Train a decision tree classifier.
    """
    if hasattr(X_train, 'toarray'):
        X_train = X_train.toarray()
    if hasattr(X_test, 'toarray'):
        X_test = X_test.toarray()
    
    # Build the tree
    tree = build_tree(
        X_train, 
        y_train, 
        max_depth=max_depth, 
        min_samples_split=min_samples_split
    )
    
    if return_predictions:
        return predict(tree, X_test)
    
    y_pred_train = predict(tree, X_train)
    y_pred_test = predict(tree, X_test)
    
    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)
    
    return test_acc, train_acc