from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint

def random_forest_classifier(X_train, X_test, y_train, y_test, return_predictions=False):
    """
    Returns:
        If return_predictions=False: (test_accuracy, train_accuracy)
        If return_predictions=True: predictions on test set
    """
    # parameter distribution for RandomizedSearchCV
    param_dist = {
        'n_estimators': randint(100, 500),
        'max_depth': [None] + list(range(10, 50, 10)),
        'min_samples_split': randint(2, 20),
        'min_samples_leaf': randint(1, 10),
        'max_features': ['sqrt', 'log2', None],
        'bootstrap': [True, False]
    }
    
    rf = RandomForestClassifier(random_state=42)
    
    random_search_rf = RandomizedSearchCV(
        rf,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )
    
    random_search_rf.fit(X_train, y_train)
    best_rf = random_search_rf.best_estimator_
    
    if return_predictions:
        return best_rf.predict(X_test)
    
    y_pred_rf = best_rf.predict(X_test)
    return accuracy_score(y_test, y_pred_rf), accuracy_score(y_train, best_rf.predict(X_train))