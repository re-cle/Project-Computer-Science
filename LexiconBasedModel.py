import os
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
import re
import emoji
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from textblob import TextBlob
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import time

# Set true to force an equal number of elements per sentiment class
equalSampleSize = False

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

# Ground truth and split data
X = df['cleaned_text']
y_true = df['label']

# Initialize VADER and measure runtime
analyzer = SentimentIntensityAnalyzer()
start_time = time.time()
# y_pred = X_test.apply(lambda x: classify_sentiment(analyzer.polarity_scores(x)['compound']))
y_pred = df['cleaned_text'].apply(lambda x: 'positive' if analyzer.polarity_scores(x)['compound'] > 0.05
                                    else 'negative' if analyzer.polarity_scores(x)['compound'] < -0.05 else 'neutral')
runtime = time.time() - start_time

# Calculate metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred, average='weighted')
recall = recall_score(y_true, y_pred, average='weighted')
f1 = f1_score(y_true, y_pred, average='weighted')

# Confusion matrix
cm = confusion_matrix(y_true, y_pred, labels=['positive', 'neutral', 'negative'])
specificity = 1 - (cm[0,1] + cm[2,1]) / (cm[0,1] + cm[2,1] + cm[1,1])
tn = cm[1, 1]
fp = cm[0, 1] + cm[2, 1]
# specificity = 1 - (fp / (fp + tn))
metrics = {'Accuracy': accuracy, 'Precision': precision, 'Recall': recall, 'Specificity': specificity, 'F1-Score': f1}

# Confusion Matrix Heatmap
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="YlGnBu", xticklabels=['positive', 'neutral', 'negative'],
            yticklabels=['positive', 'neutral', 'negative'])
plt.title('Confusion Matrix Heatmap for Sentiment Predictions')
plt.xlabel('Predicted Sentiment')
plt.ylabel('Actual Sentiment')
plt.show()

# Metrics Heatmap
metrics_df = pd.DataFrame(metrics, index=['Metrics'])
plt.figure(figsize=(8, 3))
sns.heatmap(metrics_df, annot=True, cmap="YlOrBr", fmt=".2f")
plt.title("Evaluation Metrics Heatmap")
plt.show()

# Runtime Plot
plt.figure(figsize=(6, 4))
plt.bar(['Runtime (seconds)'], [runtime])
plt.title('Runtime for Sentiment Analysis')
plt.ylabel('Time (seconds)')
plt.show()