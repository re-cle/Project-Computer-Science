import re
import emoji
import pandas as pd
import numpy as np
import tf_keras
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, TFDistilBertModel
from tf_keras.models import Model
from tf_keras.layers import Input, LSTM, Dense, Dropout
from tf_keras.optimizers import Adam
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob
import time
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score

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

# Prepare labels: convert sentiments to numbers (positive: 2, neutral: 1, negative: 0)
#   => labels have to be >= 0
X = df['cleaned_text']
X = X.fillna('')
y = df['airline_sentiment'].map({'positive': 2, 'neutral': 1, 'negative': 0}).values
# Split the dataset into training and testing sets
X_train_text, X_test_text, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Tokenize using BERT tokenizer
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Use DistilBert as it is faster than normal Bert while retaining most of its performance
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

# Set the maximum sequence length (the higher, the longer the run-time)
# MAX_LEN = 100
MAX_LEN = 64

# Tokenizing the text data and ensuring consistent input lengths
def tokenize_texts(texts, tokenizer, max_len=100):
    if isinstance(texts, pd.Series):
        texts = texts.tolist()

    # Ensure each item is a string
    texts = [str(text) for text in texts]
    return tokenizer(
        texts,
        padding='max_length',  # Ensure padding to max_len
        truncation=True,       # Truncate to max_len if needed
        max_length=max_len,
        return_tensors='tf'
    )
X_train_tokenized = tokenize_texts(X_train_text, tokenizer, max_len=MAX_LEN)
X_test_tokenized = tokenize_texts(X_test_text, tokenizer, max_len=MAX_LEN)

# Load pre-trained BERT model for embedding extraction
# bert_model = TFBertModel.from_pretrained('bert-base-uncased')
# bert_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased', force_download=True)
bert_model = TFDistilBertModel.from_pretrained(
    'distilbert-base-uncased')
# Confirm the model type
print(f"Model loaded: {bert_model.name_or_path}")

# Use to deactive training for layers
#   => Reduces both run-time and quality of predictions
# for layer in bert_model.layers:
#     layer.trainable = False

# Define a custom model that combines BERT and LSTM
input_ids = Input(shape=(MAX_LEN,), dtype=tf.int32, name='input_ids')  # Shape is fixed to MAX_LEN
attention_mask = Input(shape=(MAX_LEN,), dtype=tf.int32, name='attention_mask')

# Get BERT embeddings
bert_output = bert_model(input_ids, attention_mask=attention_mask)
sequence_output = bert_output.last_hidden_state  # Get the embeddings for each token
lstm_output = LSTM(128, return_sequences=False)(sequence_output)

# Dense layers for classification
# Dropout layer to reduce overfitting
dropout = Dropout(0.3)(lstm_output)
dense = Dense(64, activation='relu')(dropout)
output = Dense(3, activation='softmax')(dense)  # Assuming 3 classes: positive, neutral, negative

# Build and compile the model
model = Model(inputs=[input_ids, attention_mask], outputs=output)
model.compile(optimizer=Adam(learning_rate=2e-5),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Summary of the model
model.summary()
# Train the model
start_train = time.time()
history = model.fit(
    {'input_ids': X_train_tokenized['input_ids'], 'attention_mask': X_train_tokenized['attention_mask']},
    y_train,
    validation_data=(
        {'input_ids': X_test_tokenized['input_ids'], 'attention_mask': X_test_tokenized['attention_mask']},
        y_test
    ),
    epochs=3,
    batch_size=64
)
end_train = time.time()
training_time = end_train - start_train


# Evaluate the model

# Predict on test set
start_test = time.time()
y_pred = model.predict({'input_ids': X_test_tokenized['input_ids'], 'attention_mask': X_test_tokenized['attention_mask']})
end_test = time.time()
testing_time = end_test - start_test
y_pred_classes = np.argmax(y_pred, axis=1)


# Identify misclassified tweets
misclassified_indices = np.where(y_test != y_pred_classes)[0]

# Create a DataFrame of misclassified tweets
misclassified_tweets = pd.DataFrame({
    'Cleaned_Text': df.loc[X_test_text.index[misclassified_indices], 'cleaned_text'].values,
    'True_Label': y_test[misclassified_indices],
    'Predicted_Label': y_pred_classes[misclassified_indices]
})

# Map numerical labels back to sentiment for better readability
label_mapping = {2: 'positive', 1: 'neutral', 0: 'negative'}
misclassified_tweets['True_Label'] = misclassified_tweets['True_Label'].map(label_mapping)
misclassified_tweets['Predicted_Label'] = misclassified_tweets['Predicted_Label'].map(label_mapping)

# Save to CSV
misclassified_tweets.to_csv('misclassified_tweets_DistilBERT.csv', index=False)


# Confusion Matrix
conf_mat = confusion_matrix(y_test, y_pred_classes)

# Calculate the percentages by normalizing the confusion matrix row-wise
conf_mat_sum = conf_mat.sum(axis=1)[:, np.newaxis]
with np.errstate(divide='ignore', invalid='ignore'):
    conf_mat_percent = np.divide(conf_mat.astype('float') * 100, conf_mat_sum)
    conf_mat_percent = np.nan_to_num(conf_mat_percent)  # Replace NaNs with 0

# Create annotations that include both count and percentage values
annot = np.empty_like(conf_mat).astype(str)
for i in range(conf_mat.shape[0]):
    for j in range(conf_mat.shape[1]):
        annot[i, j] = f'{conf_mat[i, j]}\n({conf_mat_percent[i, j]:.1f}%)'

# Plot confusion matrix with both count and percentage values
plt.figure(figsize=(6, 4))
sns.heatmap(conf_mat, annot=annot, fmt='', cmap='Blues',
            xticklabels=['Negative', 'Neutral', 'Positive'], yticklabels=['Negative', 'Neutral', 'Positive'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title(f'Confusion Matrix (Cross-Validation)')
plt.show()

# Compute Metrics
accuracy = accuracy_score(y_test, y_pred_classes)
precision = precision_score(y_test, y_pred_classes, average='macro', zero_division=0)
recall = recall_score(y_test, y_pred_classes, average='macro', zero_division=0)
f1 = f1_score(y_test, y_pred_classes, average='macro', zero_division=0)
fpr_per_class = []
for i in range(len(conf_mat)):
    fp = conf_mat[:, i].sum() - conf_mat[i, i]  # False Positives for class i
    tn = conf_mat.sum() - (
                conf_mat[i, :].sum() + conf_mat[:, i].sum() - conf_mat[i, i])  # True Negatives for class i
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0  # Handle division by zero
    fpr_per_class.append(fpr)
fpr = np.mean(fpr_per_class)  # Macro FPR

# Print and Plot Metrics
metrics_summary = {
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1-score', '1 - FPR'],
    'Value': [accuracy, precision, recall, f1, 1 - fpr]
}
df_metrics = pd.DataFrame(metrics_summary)
# Reshape df_metrics to a suitable format for a heatmap
df_metrics_heatmap = df_metrics.set_index('Metric').T

# Define a red-to-green color map
cmap = sns.color_palette("RdYlGn", as_cmap=True)

# Plotting the heatmap of performance metrics
plt.figure(figsize=(10, 6))
sns.heatmap(df_metrics_heatmap, annot=True, fmt=".4f", cmap=cmap, linewidths=0.5, vmin=0, vmax=1)
plt.title("Classifier Performance Comparison (Higher is Better)")
plt.show()

# Runtime visualization
runtime_summary = {
    'Stage': ['Training Time', 'Testing Time'],
    'Time (seconds)': [training_time, testing_time]
}
df_runtime = pd.DataFrame(runtime_summary)

plt.figure(figsize=(8, 5))
sns.barplot(data=df_runtime, x='Stage', y='Time (seconds)')
plt.title('Model Runtime')
plt.show()
