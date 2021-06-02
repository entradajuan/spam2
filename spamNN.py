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



