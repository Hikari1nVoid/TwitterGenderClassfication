#!/usr/bin/python
# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
import time
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer, \
    CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import cross_val_score
from sklearn import preprocessing, svm
from sklearn.metrics import accuracy_score, \
    precision_recall_fscore_support, roc_curve, pairwise
from sklearn.metrics import recall_score, precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier


def cleantext(s):
    s = re.sub(URL_PATTERN, ' ', s)
    s = re.sub(HASHTAG_PATTERN, ' ', s)
    s = re.sub(MENTION_PATTERN, ' ', s)
    s = re.sub('\s+', ' ', s)

    return s


def cleanname(s):
    s = re.sub(NAME_PATTERN, ' ', s)
    s = re.sub('\s+', ' ', s)
    return s


metadata = pd.read_csv('gender-classifier-DFE-791531.csv',
                       encoding='latin1', usecols=[
    0,
    5,
    6,
    10,
    14,
    19,
    ])
data = metadata.loc[metadata['gender'].isin(['female', 'male'])
                    & metadata['gender:confidence'] >= 0.6]

##clean Nan

data.description = data.description.fillna(' ')
data.text = data.text.fillna(' ')

##clean tweets

URL_PATTERN = \
    '(http:\/\/www\.|https:\/\/www\.|http:\/\/|https:\/\/)?[a-z0-9]+([\-\.]{1}[a-z0-9]+)*\.[a-z]{2,5}(:[0-9]{1,5})?(\/.*)?$'
HASHTAG_PATTERN = '#'
MENTION_PATTERN = '@\w*'
data['text_clean'] = [cleantext(s) for s in data['text']]
data['desc_clean'] = [s.lower() for s in data['description']]
NAME_PATTERN = '[^a-zA-Z]'
data['name_clean'] = [cleanname(s) for s in data['name']]

## countVectorizer

vectorizer = CountVectorizer(stop_words='english', lowercase=True,
                             ngram_range=(1, 2))
encoder = LabelEncoder()

data['all_text'] = data[['text_clean', 'desc_clean']].apply(lambda x: \
        ' '.join(x), axis=1)
x = vectorizer.fit_transform(data['all_text'])
y = encoder.fit_transform(data['gender'])
nb = MultinomialNB()
acc = []
pre = []
recall = []
duration = []
for i in range(10):
    (x_train, x_test, y_train, y_test) = train_test_split(x, y,
            test_size=0.1)
    start = time.time()
    nb.fit(x_train, y_train)
    end = time.time()
    duration.append(end - start)
    ypred = nb.predict(x_test)
    acc.append(nb.score(x_test, y_test))
    pre.append(precision_score(y_test, ypred))
    recall.append(recall_score(y_test, ypred))
print("result:\n")
print(max(acc))
print()
print(np.mean(acc))
print()
print(np.mean(pre))
print()
print(np.mean(recall))
print()
print(duration[0])
