import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
# Kernel function: Gaussian RBF
def kernel_function(X, Y=None, gamma=1.0):
    if Y is None:
        Y = X
    return rbf_kernel(X, Y, gamma=gamma)



# Generalized Kernel Fisher Discriminant for multi-class
def kernel_fisher_discriminant_multiclass(X, y, kernel_function, gamma=1.0):
    # Compute kernel matrix
    K = kernel_function(X, X, gamma=gamma)
    
    classes = np.unique(y)
    m = len(y)
    
    # Class mean vectors and scatter matrices
    overall_mean = np.mean(K, axis=0)
    Sw = np.zeros((m, m))  # Within-class scatter matrix
    Sb = np.zeros((m, m))  # Between-class scatter matrix
    
    for cls in classes:
        idx_class = np.where(y == cls)[0]
        K_class = K[idx_class, :]
        class_mean = np.mean(K_class, axis=0)
        
        # Within-class scatter
        Sw += np.dot((K_class - class_mean).T, (K_class - class_mean))
        
        # Between-class scatter
        n_cls = len(idx_class)
        Sb += n_cls * np.outer(class_mean - overall_mean, class_mean - overall_mean)
    
    # Solve generalized eigenvalue problem
    eigvals, eigvecs = np.linalg.eig(np.linalg.pinv(Sw).dot(Sb))
    
    # Sort eigenvalues and corresponding eigenvectors
    sorted_indices = np.argsort(eigvals)[::-1]
    eigvals = eigvals[sorted_indices]
    eigvecs = eigvecs[:, sorted_indices]
    
    # Select top eigenvectors (number of classes - 1)
    W = eigvecs[:, :int(len(classes) *0.7)]
    
    # Project data onto new axes
    reduced_data = K.dot(W)
    
    return reduced_data, W


# Main function
def KernalFisherDisc(X, y ):
    # Load the dataset
    # file_path = "features_3_sec.csv"  # Replace with your CSV file
   
    
    # Standardize the features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Apply Kernel Fisher Discriminant for Multi-Class
    gamma = 0.1  # RBF kernel parameter
    reduced_data, W = kernel_fisher_discriminant_multiclass(X, y, kernel_function, gamma=gamma)
    return reduced_data, W
