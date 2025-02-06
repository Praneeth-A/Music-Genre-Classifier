import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

from svm_classifier import svm_classifier
from decision_classifier import decision_tree_classifier
from knn_classifier import knn_classifier
from ensemble_classifier import ensemble_classifier

from KernalFisherDisc import KernalFisherDisc
data = pd.read_csv('features_30_sec.csv')

X = data.drop(columns=['filename','label'])  
y = LabelEncoder().fit_transform(data['label'])  # Encode the genre labels

# Handle missing or infinite values by replacing them with a valid value (e.g, mean)
X_numeric = X.select_dtypes(include=['number'])  # Select only numeric columns
X[X_numeric.columns] = X_numeric.fillna(X_numeric.mean())

# Handle non-numeric columns separately example, you can fill missing labels with a placeholder or mode:
# X['label'] = X['label'].fillna(X['label'].mode()[0])

# categorical_columns = X.select_dtypes(include=['object']).columns.tolist()  # identify categorical columns and apply OneHotEncoder if necessary

X = X.loc[:, X.nunique() > 1]                     # Drop constant columns that have single unique value
try:
    X, w = KernalFisherDisc(X, y)  
    if np.iscomplexobj(X):
        X = np.real(X)# Ensure this function is implemented correctly
except NameError:
    raise ValueError("Function KernalFisherDisc is not defined. Please define it before running this code.")
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns),
#         ('num', StandardScaler(), X.select_dtypes(include=['number']).columns.tolist())
#     ])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# X_train = preprocessor.fit_transform(X_train)
# X_test = preprocessor.transform(X_test)


def evaluate_model(model_name, classifier_func, X_train, X_test, y_train, y_test):
    
    # test and train accuracies
    test_acc, train_acc = classifier_func(X_train, X_test, y_train, y_test)
    
    y_pred = classifier_func(X_train, X_test, y_train, y_test, return_predictions=True)
    
    print(f"\n{model_name} Performance:")
    print(f"Train Accuracy: {train_acc:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("Loading...")
    
    return test_acc, train_acc

print("\n--- Model Performance Evaluation ---")
svm_test_acc, svm_train_acc = evaluate_model("SVM", svm_classifier, X_train, X_test, y_train, y_test)
dt_test_acc, dt_train_acc = evaluate_model("Decision Tree", decision_tree_classifier, X_train, X_test, y_train, y_test)
knn_test_acc, knn_train_acc = evaluate_model("KNN", knn_classifier, X_train, X_test, y_train, y_test)
ensemble_test_acc, ensemble_train_acc = evaluate_model(
    "Ensemble", ensemble_classifier, X_train, X_test, y_train, y_test
)

plt.figure(figsize=(12, 7))

models = ['KNN', 'Decision Tree', 'SVM', 'Ensemble']
train_accuracies = [svm_train_acc, dt_train_acc, knn_train_acc, ensemble_train_acc]
test_accuracies = [svm_test_acc, dt_test_acc, knn_test_acc, ensemble_test_acc]

x = np.arange(len(models))
width = 0.35

bars1 = plt.bar(x - width/2, train_accuracies, width, label='Train Accuracy', color='#4CAF50', edgecolor='black', linewidth=1.2)
bars2 = plt.bar(x + width/2, test_accuracies, width, label='Test Accuracy', color='#FF5733', edgecolor='black', linewidth=1.2)

plt.xlabel('Models', fontsize=14, fontweight='bold', color='#333333')
plt.ylabel('Accuracy (%)', fontsize=14, fontweight='bold', color='#333333')
plt.title('Model Performance Comparison', fontsize=16, fontweight='bold', color='#333333')

plt.xticks(x, models, fontsize=12, fontweight='bold', color='#333333')
plt.yticks(fontsize=12, color='#333333')
plt.legend(loc='upper left', fontsize=12, title='Accuracy Types', title_fontsize=13)
for bar in bars1:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')

for bar in bars2:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 0.01, round(yval, 2), ha='center', va='bottom', fontsize=11, fontweight='bold', color='#333333')
plt.tight_layout()
plt.savefig('model_performance_comparison.png', dpi=300)
plt.close()


print("\n--- Final Accuracy Comparison ---")
print("SVM: Train Accuracy:", svm_train_acc, "Test Accuracy:", svm_test_acc)
print("Decision Tree: Train Accuracy:", dt_train_acc, "Test Accuracy:", dt_test_acc)
print("KNN: Train Accuracy:", knn_train_acc, "Test Accuracy:", knn_test_acc)
print("Ensemble: Train Accuracy:", ensemble_train_acc, "Test Accuracy:", ensemble_test_acc)