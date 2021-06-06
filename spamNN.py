%tensorflow_version 2.x

import tensorflow as tf
#from tf.keras.models import Sequential
#from tf.keras.layers import Dense
import os
import io

tf.__version__

#path_to_zip = tf.keras.utils.get_file("smsspamcollection.zip",
#                  origin="https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
#                  extract=True)

#print(path_to_zip)
#print(type(path_to_zip))

#!unzip $path_to_zip -d data

#lines = io.open('/content/spam2/data/SMSSpamCollection').read().strip().split('\n')
lines = io.open('data/SMSSpamCollection').read().strip().split('\n')
print(lines)
print(type(lines))
print(lines[0])

data = []
count = 0
for e in lines:
  label, text = e.split('\t')
  if (label.lower().strip() =='spam'):
    data.append((1, text.strip()))
  else:
    data.append((0, text.strip()))

print(data)
print(len(data))
print(data[0][1])
print(type(data[0]))

import pandas as pd
df = pd.DataFrame(data, columns=['spam', 'text'])
print(df.head())
print(df.shape)

import re

def message_length(x):
  return len(x)

def num_capitals(x):
  _, count = re.subn(r'[A-Z]' , '', x)
  return count

cap_count = num_capitals('Adsd Aggggg')
print(cap_count)

def num_punctuation(x):
  _, num =  re.subn(r'\W', '', x)
  return num

df['long'] = df['text'].apply(message_length)
df['caps'] = df['text'].apply(num_capitals)
df['punct'] = df['text'].apply(num_punctuation)

print(df.head().to_string)

train = df.sample(frac=0.8, random_state=42)
test = df.drop(train.index)

print()
print(train.describe())

def make_model(inputs=3, num_units=12):
  model = tf.keras.Sequential()
  model.add(tf.keras.layers.Dense(num_units, input_dim=inputs, activation='relu'))
  model.add(tf.keras.layers.Dense(1, activation='relu'))
  model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
  return model

x_train = train[['long', 'punct', 'caps']]
y_train = train[['spam']]
print(y_train)
print(y_train.shape)
print(y_train['spam'].sum())
print(type(y_train))


x_test = test[['long', 'punct', 'caps']]
y_test = test[['spam']]

model = make_model(inputs=3, num_units=12)
print(type(model))
model.fit(x_train, y_train, epochs=10, batch_size=10)

model.evaluate(x_test, y_test)
y_train_pred = model.predict(x_train)
print(y_train_pred)
print(tf.math.confusion_matrix(tf.constant(y_train.spam), y_train_pred))

y_test_pred = model.predict_classes(x_test)
print(y_test_pred)
print(tf.math.confusion_matrix(tf.constant(y_test.spam), y_test_pred))

import stanza
#snlp = stanza.download('en') 
en = stanza.Pipeline(lang='en', processors='tokenize')


def word_counts(x, pipeline=en):
  docu = pipeline(x)
  count = sum([len(sen.tokens) for sen in docu.sentences])
  return count

df['words'] = df['text'].apply(word_counts)

train['words'] = train['text'].apply(word_counts)
test['words'] = test['text'].apply(word_counts)
x_train = train[['text', 'long', 'caps', 'punct', 'words']]
y_train = train[['spam']]

x_test = test[['text', 'long', 'caps', 'punct', 'words']]
y_test = test[['spam']]

model = make_model(inputs=4)

model.fit(x_train, y_train, epochs=20, batch_size=100)
model.evaluate(x_test, y_test)
