import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# get data
data = pd.read_csv('tweet_emotions.csv')

print(f"Number of entries before filtering: {len(data)}")

#filter data
data = data[data['sentiment'].isin(['happiness', 'neutral'])]

print(f"Number of entries after filtering: {len(data)}")

# preprocess data
nltk.download('punkt')
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

def preprocess(content):
    tokens = word_tokenize(content.lower())
    filtered_tokens = [w for w in tokens if w.isalnum() and not w in stop_words]
    return " ".join(filtered_tokens)

data['clean_text'] = data['content'].apply(preprocess)

# feature extraction
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(data['clean_text'])
y = data['sentiment'].apply(lambda x: 1 if x == 'happiness' else 0)

# train the model

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression()
model.fit(X_train, y_train)

# Evaluate the Model
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))