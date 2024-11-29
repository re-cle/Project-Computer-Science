# Import libraries for cross-validation and metrics
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time

# Define cross-validation setup
kf = StratifiedKFold(n_splits=5)

for classifier_name, classifier in classifiers.items():
    # Arrays to collect metrics and predictions
    all_y_true = []
    all_y_pred = []

    # Lists to store fold metrics
    accuracy_scores = []
    precision_scores = []
    recall_scores = []
    fpr_scores = []
    f1_scores = []
    training_times = []
    testing_times = []

    # Perform cross-validation manually
    for train_index, test_index in kf.split(X_vec, y):
        X_train, X_test = X_vec[train_index], X_vec[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # Train and predict while measuring the time
        start_train = time.time()
        classifier.fit(X_train, y_train)
        end_train = time.time()
        training_times.append(end_train - start_train)

        start_test = time.time()
        y_pred = classifier.predict(X_test)
        end_test = time.time()
        testing_times.append(end_test - start_test)

        # Collect predictions and true values
        all_y_true.extend(y_test)
        all_y_pred.extend(y_pred)

        # Calculate metrics for this fold
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, average='macro'))
        recall_scores.append(recall_score(y_test, y_pred, average='macro'))
        f1_scores.append(f1_score(y_test, y_pred, average='macro'))

        # Compute confusion matrix for the current fold
        conf_mat = confusion_matrix(y_test, y_pred)

        # Calculate FPR for each class and then average (macro FPR)
        fpr_per_class = []
        for i in range(len(conf_mat)):
            fp = conf_mat[:, i].sum() - conf_mat[i, i]  # False Positives for class i
            tn = conf_mat.sum() - (conf_mat[i, :].sum() + conf_mat[:, i].sum() - conf_mat[i, i])  # True Negatives for class i
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Handle division by zero
            fpr_per_class.append(fpr)
            
        fpr_scores.append(np.mean(fpr_per_class))  # Macro FPR for this fold


# Compute and display the average metrics across folds
print(f"Results for {classifier_name}:")
print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
print(f"Precision (macro): {np.mean(precision_scores):.4f}")
print(f"Recall (macro): {np.mean(recall_scores):.4f}")
print(f"False Positive Rate (macro): {np.mean(fpr_scores):.4f}")
print(f"F1-score (macro): {np.mean(f1_scores):.4f}")
print(f"Total Training Time: {np.sum(training_times):.4f} seconds")
print(f"Total Testing Time: {np.sum(testing_times):.4f} seconds")
