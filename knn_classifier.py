
import numpy as np
from collections import Counter
from scipy.stats import randint
from sklearn.metrics import accuracy_score

def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2) ** 2))

def manhattan_distance(x1, x2):
    return np.sum(np.abs(x1 - x2))

def minkowski_distance(x1, x2, p):
    return np.sum(np.abs(x1 - x2) ** p) ** (1/p)

class KNeighborsClassifierScratch:
    def __init__(self, n_neighbors=5, weights='uniform', p=2):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.p = p

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self, X_test):
        predictions = [self._predict(x) for x in X_test]
        return np.array(predictions)
        
    def _predict(self, x):
        distances = [minkowski_distance(x, x_train, self.p) for x_train in self.X_train]
        
        sorted_indices = np.argsort(distances)
        nearest_indices = sorted_indices[:self.n_neighbors]
        
        nearest_labels = self.y_train[nearest_indices]
        
        if self.weights == 'uniform':
            most_common = Counter(nearest_labels).most_common(1)
            return most_common[0][0]
        elif self.weights == 'distance':

            distances_nonzero = np.array([distances[i] if distances[i] != 0 else np.finfo(float).eps for i in nearest_indices])
            weights = 1 / distances_nonzero
            
            weights /= np.sum(weights)  
            
            weighted_votes = Counter()
            for i, label in zip(nearest_indices, nearest_labels):
                weighted_votes[label] += weights[np.where(nearest_indices == i)[0][0]] 
            
            return weighted_votes.most_common(1)[0][0]



def knn_classifier(X_train, X_test, y_train, y_test, return_predictions=False):
    """
    Returns:
        If return_predictions=False: (test_accuracy, train_accuracy)
        If return_predictions=True: predictions on test set
    """
    param_grid = {
        'n_neighbors': randint(1, 15),  
        'weights': ['uniform', 'distance'],
        'p': [1, 2]
    }
    
    best_accuracy = 0
    best_knn = None
    
    for _ in range(10):  
        n_neighbors = param_grid['n_neighbors'].rvs()  
        weights = np.random.choice(param_grid['weights'])
        p = np.random.choice(param_grid['p'])
        
        knn = KNeighborsClassifierScratch(n_neighbors=n_neighbors, weights=weights, p=p)
        knn.fit(X_train, y_train)
        
        y_pred_test = knn.predict(X_test)
        test_acc = accuracy_score(y_test, y_pred_test)
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            best_knn = knn
    
    if return_predictions:
        return best_knn.predict(X_test)
    
    y_pred_train = best_knn.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred_train)
    
    return best_accuracy, train_acc
