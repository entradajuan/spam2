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
print(train.index)


print(df.shape)
print(train.shape)
print(test.shape)



