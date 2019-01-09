import numpy as np
import pandas as pd
import re
import nltk
#nltk.download('stopwords')
#nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


df = pd.read_csv('imdb_labelled.txt', delimiter = '\t', quoting = 3)
print(df.head(6))
corpus = []

for i in range(0, 1000):
  review = re.sub('[^a-zA-Z]', ' ', df['Review'][i])
  review = review.lower()
  review = review.split()
  lemmatizer = WordNetLemmatizer()
  review = [lemmatizer.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
  review = ' '.join(review)
  corpus.append(review)

print("\nlenght of corpus",len(corpus))

from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(max_features = 2000)

format = lambda x: '%d' % x

X = cv.fit_transform(corpus).toarray()

print(pd.DataFrame(X, columns=cv.get_feature_names()).head())

y = df.iloc[:-1, 1].bfill().apply(format)

#y = df["Status"].values
print("\nlenght of status", y.count())

#y = cv.get_feature_names()
#print(y.tolist())

from sklearn.model_selection import train_test_split

X_train, X_test , y_train, y_test = train_test_split(X, y, test_size = 0.20)

from sklearn.naive_bayes import GaussianNB

classifier = GaussianNB()

classifier.fit(X_train, y_train)

predictions = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, predictions)

print(cm)

#from sklearn.feature_extraction.text import TfidfVectorizer

#tfidfVectorizer = TfidfVectorizer(max_features =2000)

#X = tfidfVectorizer.fit_transform(corpus).toarray()

print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(classifier.score(X_test, y_test)))