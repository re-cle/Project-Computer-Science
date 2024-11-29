# Import required libraries
import nltk
import re
import emoji
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from textblob import TextBlob

# Download nltk stopwords and other required packages
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Function to clean and correct spelling in tweet text
def clean_text(text, apply_stemming=False, apply_lemmatization=True, fix_spelling_errors=True):
    # Initialize stemmer and lemmatizer
    stemmer = PorterStemmer()
    lemmatizer = WordNetLemmatizer()

    # Remove unwanted symbols
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'[^a-zA-Z]+', ' ', text)

    # Convert hashtags, emojis, and elongated words
    hashtags = re.findall(r"#(\w+)", text)
    text += " " + " ".join([f"hashtag_{tag}" for tag in hashtags])
    text = emoji.demojize(text)
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
