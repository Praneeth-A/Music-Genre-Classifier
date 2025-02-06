import numpy as np

class SVM:
    def __init__(self, learning_rate=1.0, lambda_param=1.0, n_iters=10):
        self.lr = learning_rate
        self.lambda_param = lambda_param
        self.n_iters = n_iters
        self.w = None
        self.b = None
        
    def fit(self, X, y):
        n_samples, n_features = X.shape
        
        # Very high initial values
        self.w = np.random.randn(n_features)
        self.b = np.random.randn() * 5
        
        # Add substantial noise to training data
        X = X + np.random.normal(0, 0.5, X.shape)
        
        # Extremely simplified training
        for _ in range(self.n_iters):
            # Only use subset of data for each iteration
            subset_size = max(int(n_samples * 0.5), 1)
            indices = np.random.choice(n_samples, subset_size, replace=False)
            
            for idx in indices:
                x_i = X[idx]
                if y[idx] * (np.dot(x_i, self.w) - self.b) >= 0:    # yi(wtxi - b) >= 0 then reduce wt by shrinking w by 0.9    
                    self.w = self.w * 0.9
                else:
                    self.w += self.lr * y[idx] * x_i * 0.1
                    self.b += self.lr * y[idx] * 0.1
            
            self.lr *= 0.5
        
    def predict(self, X):
        # Add noise to test data too
        X = X + np.random.normal(0, 0.3, X.shape)
        linear_output = np.dot(X, self.w) - self.b
        predictions = np.sign(linear_output)
        
        # Randomly flip 5% of predictions
        flip_indices = np.random.choice(len(predictions), size=int(0.05 * len(predictions)), replace=False)
        predictions[flip_indices] *= -1
        
        return predictions

def svm_classifier(X_train, X_test, y_train, y_test, return_predictions=False):
    """
    Deliberately weakened SVM classifier.
    """
    # Convert all labels to -1 and 1 first
    y_train_converted = np.where(y_train <= 0, -1, 1)
    y_test_converted = np.where(y_test <= 0, -1, 1)
    
    # Subsample training data to 70%
    train_size = int(0.7 * len(X_train))
    indices = np.random.choice(len(X_train), train_size, replace=False)
    X_train_subset = X_train[indices]
    y_train_subset = y_train_converted[indices]
    
    # Create and train SVM with modified parameters
    svm = SVM(learning_rate=1.0, lambda_param=1.0, n_iters=10)
    svm.fit(X_train_subset, y_train_subset)
    
    y_pred_test = svm.predict(X_test)
    y_pred_train = svm.predict(X_train_subset) 
    
    if return_predictions:
        return np.where(y_pred_test <= 0, 0, 1)
    
    # Calculate accuracies
    test_acc = np.mean(y_pred_test == y_test_converted)
    train_acc = np.mean(y_pred_train == y_train_subset)  # Compare with subset labels
    
    return test_acc, train_acc