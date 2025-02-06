from sklearn.ensemble import VotingClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from scipy.stats import randint, uniform


def ensemble_classifier(X_train, X_test, y_train, y_test, return_predictions=False):

    svm = SVC(probability=True, random_state=42)  
    dt = DecisionTreeClassifier(random_state=42)
    knn = KNeighborsClassifier()

    param_dist = {
        'svm__C': uniform(0.1, 10),
        'svm__kernel': ['linear', 'rbf'],
        'dt__max_depth': [None] + list(range(5, 30, 5)),
        'dt__min_samples_split': randint(2, 20),
        'dt__min_samples_leaf': randint(1, 10),
        'knn__n_neighbors': randint(1, 15),
        'knn__weights': ['uniform', 'distance'],
        'voting': ['soft']
    }

    ensemble = VotingClassifier(
        estimators=[
            ('svm', svm),
            ('dt', dt),
            ('knn', knn)
        ],
        voting='soft'  
    )

    # RandomizedSearchCV
    random_search = RandomizedSearchCV(
        ensemble,
        param_distributions=param_dist,
        n_iter=50,
        cv=5,
        scoring='accuracy',
        n_jobs=-1,
        random_state=42,
        verbose=1
    )

    print("Training ensemble model...")
    random_search.fit(X_train, y_train)
    best_ensemble = random_search.best_estimator_
    
    print("\nBest parameters found:")
    for param, value in random_search.best_params_.items():
        print(f"{param}: {value}")

    if return_predictions:
        return best_ensemble.predict(X_test)

    y_pred = best_ensemble.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    train_acc = accuracy_score(y_train, best_ensemble.predict(X_train))

    print("\nIndividual Classifier Performances:")
    for name, clf in best_ensemble.named_estimators_.items():
        y_pred_individual = clf.predict(X_test)
        acc = accuracy_score(y_test, y_pred_individual)
        print(f"{name.upper()} Test Accuracy: {acc:.4f}")

    return test_acc, train_acc