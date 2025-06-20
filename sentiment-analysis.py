# Import libraries
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelBinarizer
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from wordcloud import WordCloud, STOPWORDS
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize
import spacy
import re, string, unicodedata
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.stem import LancasterStemmer, WordNetLemmatizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from textblob import TextBlob
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from bs4 import BeautifulSoup
nltk.download('stopwords')

# Tokenizer and stopwords
tokenizer = ToktokTokenizer()
stopwords = nltk.corpus.stopwords.words('english')

# Noise removal function
def noiseremoval_text(text):
    if not isinstance(text, str):
        return text
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = re.sub('\\[[^]]*\\]', '', text)
    return text

# Stemming function
def stemmer(text):
    ps = nltk.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text

# Stopword removal function
def removing_stopwords(text, is_lower_case=False):
    tokens = tokenizer.tokenize(text)
    tokens = [i.strip() for i in tokens]
    if is_lower_case:
        filtered_tokens = [i for i in tokens if token not in stopwords]
    else:
        filtered_tokens = [i for i in tokens if i.lower() not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

# Apply preprocessing
data['review'] = data['review'].apply(noiseremoval_text)
data['review'] = data['review'].apply(stemmer)
data['review'] = data['review'].apply(removing_stopwords)

# Split data
train_reviews_data = data.review[:30000]
test_review_data = data.review[30000:]
train_data = data.sentiment[:30000]
test_data = data.sentiment[30000:]

# Vectorization
cv = CountVectorizer(min_df=0.0, max_df=1.0, binary=False, ngram_range=(1,3))
cv_train = cv.fit_transform(train_reviews_data)
cv_test = cv.transform(test_review_data)

tf = TfidfVectorizer(min_df=0.0, max_df=1.0, use_idf=True, ngram_range=(1,3))
tf_train = tf.fit_transform(train_reviews_data)
tf_test = tf.transform(test_review_data)

# Label encoding
label = LabelBinarizer()
sentiment_data = label.fit_transform(data['sentiment'])

# Model training
logistic = LogisticRegression(penalty='l2', max_iter=500, C=1, random_state=42)
lr_bow = logistic.fit(cv_train, train_data)
pred = logistic.predict(cv_test)

# Accuracy
accuracy = accuracy_score(test_data, pred)
print(accuracy)  # Output: 0.6984
