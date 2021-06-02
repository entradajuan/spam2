%tensorflow_version 2.x
import tensorflow as tf
#from tf.keras.models import Sequential
#from tf.keras.layers import Dense
import os
import io

tf.__version__

path_to_zip = tf.keras.utils.get_file("smsspamcollection.zip",
                  origin="https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip",
                  extract=True)

print(path_to_zip)
print(type(path_to_zip))