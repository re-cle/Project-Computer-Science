# Import required libraries
import os.path
import matplotlib
import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import emoji
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
# Import classifiers
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import time

# Set to True to force an equal number of elements per sentiment class
equalSampleSize = False
# Set to True to apply hyperparameter tuning to Logistic Regression
lr_HyperparameterTuning = False
# Set to True to use the previously identified best parameter combination for Logistic Regression (overwrites lr_HyperparameterTuning)
useBestLrParams = False
if useBestLrParams:
    lr_HyperparameterTuning = False
# Set to True to use balanced class weights for Logistic Regression
# Only has effect if useBestLrParams is True
useBalancedClassWeights = False
# Only use Logistic Regression as a Classifier
only_LR = False

# Ensures that the visualization works
matplotlib.use('TkAgg')

# Download nltk stopwords and other required packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Perform preprocessing and save the results in a separate .csv file
if not os.path.exists('preprocessed_tweets_improved.csv'):
    # Load the dataset (update with your actual CSV file name)
    data = pd.read_csv("Tweets.csv")

    # Select relevant columns (text, airline_sentiment, and airline_sentiment_confidence)
    df = data[['text', 'airline_sentiment', 'airline_sentiment_confidence']]

    # Filter data where sentiment confidence is high (e.g., greater than 0.5)
    df = df[df['airline_sentiment_confidence'] > 0.5]

    # Function to clean and correct spelling in tweet text
    def clean_text(text, apply_stemming=False, apply_lemmatization=True, fix_spelling_errors=True):
        # Initialize stemmer and lemmatizer
        stemmer = PorterStemmer()
        lemmatizer = WordNetLemmatizer()
        # Remove URLs
        text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
        # Remove mentions (@username)
        text = re.sub(r'@\w+', '', text)
        # Remove punctuations and numbers
        text = re.sub(r'[^a-zA-Z]+', ' ', text)
        # Remove hashtags but keep their text
        hashtags = re.findall(r"#(\w+)", text)
        text += " " + " ".join([f"hashtag_{tag}" for tag in hashtags])
        # Replace emojis with descriptions
        text = emoji.demojize(text)
        # Normalize elongated words (e.g., "soooo" to "so")
        text = re.sub(r'(.)\1{2,}', r'\1\1', text)
        # Convert to lowercase
        text = text.lower()
        # Tokenize
        words = word_tokenize(text)
        # Remove stopwords
        words = [word for word in words if word not in stopwords.words('english')]

        # Apply stemming or lemmatization based on flags
        if apply_stemming:
            words = [stemmer.stem(word) for word in words]
        if apply_lemmatization:
            words = [lemmatizer.lemmatize(word) for word in words]

        # Rejoin cleaned words
        cleaned_text = ' '.join(words)

        if fix_spelling_errors:
            # Correct spelling using TextBlob
            corrected_text = str(TextBlob(cleaned_text).correct())
            return corrected_text
        else:
            return cleaned_text

    # Apply text cleaning and spelling correction function to dataset
    df['cleaned_text'] = df['text'].apply(clean_text)

    # Convert sentiment labels to numerical format
    df['label'] = df['airline_sentiment'].map({'positive': 1, 'neutral': 0, 'negative': -1})


    # Save the dataframe to a CSV file
    df.to_csv('preprocessed_tweets_improved.csv', index=False)

# If the file with the preprocessed tweets already exists, read it in instead of performing preprocessing again
#   => Used to save run-time
else:
    df = pd.read_csv('preprocessed_tweets_improved.csv')

# Sample an equal number of tweets from each class
if equalSampleSize:
    # Determine the minimum count among the classes
    min_count = df['airline_sentiment'].value_counts().min()
    # For each class, only use the amount of elements available for the smallest class
    df = df.groupby('airline_sentiment').apply(lambda x: x.sample(min_count, random_state=42)).reset_index(drop=True)

# Display the balanced dataset's class distribution
print(df['airline_sentiment'].value_counts())

# Split the dataset into input features and labels
X = df['cleaned_text']
y = df['label']
X = X.fillna('')
print ('X', X)

print('y', y)

print('Data Cleaning Complete')

# Convert text data to numerical format using TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X_vec = vectorizer.fit_transform(X)
y = y.values

# Define classifiers to compare
if only_LR:
    classifiers = {
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr'),
    }
else:
    classifiers = {
        'Naive Bayes': MultinomialNB(),
        'Support Vector Machine': SVC(kernel='linear'),  # Linear kernel is often best for text classification
        'Logistic Regression': LogisticRegression(max_iter=1000, multi_class='ovr'),  # Increase iterations for convergence
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }

# Specify scoring metrics
scoring = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']

# Define class labels for the heatmap
class_labels = ['Negative', 'Neutral', 'Positive']

# Define cross-validation setup
kf = StratifiedKFold(n_splits=5)

# Initialize a dictionary to store the metrics for each classifier
metrics_summary = {
    'Classifier': [],
    'Accuracy': [],
    'Precision': [],
    'Recall': [],
    'F1-score': [],
    '1 - FPR': []
}
times_summary = {
    'Classifier': [],
    'Avg Training Time (s)': [],
    'Avg Testing Time (s)': []
}

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


    # If lr_HyperparameterTuning is True, perform hyperparameter tuning for Logistic Regression
    if classifier_name == 'Logistic Regression' and lr_HyperparameterTuning:
        # Logistic Regression Tuning
        # lr_param_grid = {'C': [0.01, 0.1, 1, 10, 100], 'penalty': ['l2'], 'solver': ['lbfgs']}
        param_grid = {
            'C': [0.01, 0.1, 1, 10, 100],  # Regularization strength
            'penalty': ['l2'],  # Regularization types
            'solver': ['lbfgs'],  # Solvers to try
            # 'max_iter': [100, 500, 1000, 10000],  # Iterations
            'tol': [1e-10, 1e-4, 1e-3, 1e-2]  # Convergence tolerance
        }
        # lr_grid_search = GridSearchCV(LogisticRegression(max_iter=1000, multi_class='ovr'), lr_param_grid, cv=5)
        grid_search = GridSearchCV(
            # LogisticRegression(random_state=42, multi_class='ovr'),
            LogisticRegression(random_state=42, multi_class='ovr', max_iter=10000),
            param_grid,
            cv=5,  # 5-fold cross-validation
        )
        grid_search.fit(X_vec, y)
        print("Best Logistic Regression Parameters:", grid_search.best_params_)
        best_lr = grid_search.best_estimator_
        classifier = best_lr
        # classifier = LogisticRegression(random_state=42, multi_class='ovr', max_iter=10000, C=10, tol=1e-10)

    # If useBestLrParams is True, use the previously found parameter combination to perform Logistic Regression
    if classifier_name == 'Logistic Regression' and useBestLrParams:
        if useBalancedClassWeights:
            classifier = LogisticRegression(random_state=42, multi_class='ovr', max_iter=10000, C=10, tol=1e-10,
                                            class_weight='balanced')
        else:
            classifier = LogisticRegression(random_state=42, multi_class='ovr', max_iter=10000, C=10, tol=1e-10)

    # Perform cross-validation manually
    for train_index, test_index in kf.split(X_vec, y):
        print('Test_index', test_index)
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
            tn = conf_mat.sum() - (
                        conf_mat[i, :].sum() + conf_mat[:, i].sum() - conf_mat[i, i])  # True Negatives for class i
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Handle division by zero
            fpr_per_class.append(fpr)
        fpr_scores.append(np.mean(fpr_per_class))  # Macro FPR for this fold

    # Append the average metrics across folds to the metrics summary dictionary
    metrics_summary['Classifier'].append(classifier_name)
    metrics_summary['Accuracy'].append(np.mean(accuracy_scores))
    metrics_summary['Precision'].append(np.mean(precision_scores))
    metrics_summary['Recall'].append(np.mean(recall_scores))
    metrics_summary['F1-score'].append(np.mean(f1_scores))
    metrics_summary['1 - FPR'].append(1 - np.mean(fpr_scores))

    # Append the average times across folds to the times summary dictionary
    times_summary['Classifier'].append(classifier_name)
    times_summary['Avg Training Time (s)'].append(np.mean(training_times))
    times_summary['Avg Testing Time (s)'].append(np.mean(testing_times))


    # Compute and display the average metrics across folds
    print(f"Results for {classifier_name}:")
    print(f"Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Precision (macro): {np.mean(precision_scores):.4f}")
    print(f"Recall (macro): {np.mean(recall_scores):.4f}")
    print(f"False Positive Rate (macro): {np.mean(fpr_scores):.4f}")
    print(f"F1-score (macro): {np.mean(f1_scores):.4f}")
    print(f"Total Training Time: {np.sum(training_times):.4f} seconds")
    print(f"Total Testing Time: {np.sum(testing_times):.4f} seconds")
    print()

    # Compute confusion matrix based on combined cross-validated predictions
    conf_mat = confusion_matrix(all_y_true, all_y_pred)

    # Calculate the percentages by normalizing the confusion matrix row-wise
    conf_mat_percent = conf_mat.astype('float') / conf_mat.sum(axis=1)[:, np.newaxis] * 100

    # Create annotations that include both count and percentage values
    annot = np.empty_like(conf_mat).astype(str)
    for i in range(conf_mat.shape[0]):
        for j in range(conf_mat.shape[1]):
            annot[i, j] = f'{conf_mat[i, j]}\n({conf_mat_percent[i, j]:.1f}%)'

    # Plot confusion matrix with both count and percentage values
    plt.figure(figsize=(6, 4))
    sns.heatmap(conf_mat, annot=annot, fmt='', cmap='Blues',
                xticklabels=class_labels, yticklabels=class_labels)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{classifier_name} Confusion Matrix (Cross-Validation)')
    plt.show()

    # Save misclassified tweets for further analysis
    misclassified_tweets = []
    label_mapping = {1: 'positive', 0: 'neutral', -1: 'negative'}
    for true, pred, text in zip(all_y_true, all_y_pred, X):
        if true != pred:
            misclassified_tweets.append(
                {'Cleaned_Text': text, 'True_Label': label_mapping[true], 'Predicted_Label': label_mapping[pred]})

    misclassified_df = pd.DataFrame(misclassified_tweets)
    fileName = 'misclassified_tweets_' + str(classifier_name) + '.csv'
    misclassified_df.to_csv(fileName, index=False)


# Convert the metrics and times summaries to DataFrames
metrics_df = pd.DataFrame(metrics_summary).set_index('Classifier')
times_df = pd.DataFrame(times_summary).set_index('Classifier')

# Define a red-to-green color map
cmap = sns.color_palette("RdYlGn", as_cmap=True)

# Plotting the heatmap of performance metrics
plt.figure(figsize=(10, 6))
sns.heatmap(metrics_df, annot=True, fmt=".4f", cmap=cmap, linewidths=0.5, vmin=0, vmax=1)
plt.title("Classifier Performance Comparison (Higher is Better)")
plt.show()

# Define a green-to-red colormap that quickly transitions to red
runtime_cmap = sns.color_palette("RdYlGn_r", as_cmap=True, n_colors=10)

# Plotting the heatmap for runtime metrics with adjusted color intensity
plt.figure(figsize=(6, 4))
sns.heatmap(times_df, annot=True, fmt=".4f", cmap=runtime_cmap, linewidths=0.5)
plt.title("Classifier Runtime Comparison (Lower is Better)")
plt.show()