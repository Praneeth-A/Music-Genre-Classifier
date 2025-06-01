# Music Genre Classifier

This project implements an ensemble model integrating SVM, Decision Trees, and KNN through soft voting for genre classification of 3-second and 30-second audio tracks. Incorporated Kernel Fisher Discriminant for non-linear dimensionality reduction to optimize input data and model performance. 


## Features

- **Multiple Classifiers:** SVM, Decision Tree, KNN, Random Forest, Ensemble.
- **Performance Evaluation:** Compares accuracy and displays confusion matrices.
- **Data Preprocessing:** Handles missing values and encodes labels.
- **Visualization:** Plots model performance and confusion matrices.

## Project Structure

- `main.py` — Main script for data loading, preprocessing, training, and evaluation.
- `svm_classifier.py` — SVM classifier implementation.
- `decision_classifier.py` — Decision Tree classifier implementation.
- `knn_classifier.py` — KNN classifier implementation.
- `ensemble_classifier.py` — Random Forest implementation.
- `KernalFisherDisc.py` — Kernel Fisher Discriminant implementation.
- `features_30_sec.csv`, `features_3_sec.csv` — Feature datasets.
- `confusion_matrix_*.png`, `model_performance_comparison_*.png` — Output plots.
- `requirements.txt` — Python dependencies.

## Installation

1. Clone the repository.
2. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage

1. Place the feature CSV files in the project directory.
2. Run the main script:
    ```sh
    python main.py
    ```
3. View the output plots for model performance and confusion matrices.

## Customization

- To use a different feature file, change the filename in `main.py`:
    ```python
    data = pd.read_csv('features_30_sec.csv')
    ```
- Adjust classifier parameters in their respective files for experimentation.

## Results

- Model performance and confusion matrices are saved as PNG files in the project directory.
- Compare the effectiveness of different classifiers visually.

## Dependencies

See [requirements.txt](requirements.txt) for a full list. Key libraries include:
- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`

## License

This project is for educational purposes.
